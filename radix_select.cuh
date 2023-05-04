#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

// 32-bit unsigned integer
typedef unsigned int uint32_cu;

__device__ uint32_cu positionBits(uint32_cu value, int position) {
  return (value >> ((sizeof(uint32_cu) - position) * 8)) & 0xff;
}

__global__ void collectHistogram(int numValues, uint32_cu *values,
                                 int *histogram, int position) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < numValues; i += stride) {
    uint32_cu bin = positionBits(values[i], position);
    atomicAdd(&histogram[bin], 1);
  }
}

/*
  This histogram collector uses shared memory. This is a performance improvement
  only for when numValues is very large (e.g., this is only used in the first
  iteration).
*/
__global__ void collectHistogramSharedMem(int numValues, uint32_cu *values,
                                          int *histogram, int position) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int id = threadIdx.x;

  // first collect in per-block shared memory
  __shared__ int sharedHistogram[256];

  if (id < 256) {
    sharedHistogram[id] = 0; // need to zero out first
  }

  __syncthreads();

  for (int i = idx; i < numValues; i += stride) {
    uint32_cu bin = positionBits(values[i], position);
    atomicAdd(&sharedHistogram[bin], 1);
  }

  __syncthreads();

  // now, use 256 threads per block to add shared histogram to global histogram
  if (id < 256) {
    atomicAdd(&histogram[id], sharedHistogram[id]);
  }
}

// predicates for `thrust::copy_if`
struct belongsToPivotBin {
  int position;
  uint32_cu pivot;

  belongsToPivotBin(int position, uint32_cu pivot)
      : position(position), pivot(pivot) {}

  __device__ bool operator()(uint32_cu value) {
    return positionBits(value, position) == pivot;
  }
};
struct valueBelowThreshold {
  uint32_cu *values;
  uint32_cu threshold;

  valueBelowThreshold(uint32_cu *values, uint32_cu threshold)
      : values(values), threshold(threshold) {}

  __device__ bool operator()(uint32_cu value) { return value <= threshold; }
};

uint32_cu radix_select(uint32_cu *values, int *keys, int numValues, int k,
                       uint32_cu *kSmallestValues, int *kSmallestKeys) {
  int blockSize = 512;
  int numBlocks = (numValues + blockSize - 1) / blockSize;

  // allocate histogram, prefix sum, keys, and temporary arrays
  int *histogram, *prefixSums, *deviceKSmallestKeys;
  // TODO: cut down on size of these
  uint32_cu *tempValues1, *tempValues2;

  cudaMalloc(&histogram, 256 * sizeof(int));
  cudaMalloc(&prefixSums, 256 * sizeof(int));
  // NOTE: size of `deviceKSmallestKeys` can be reduced to `k * sizeof(int)` if
  // values are guaranteed to be unique
  // TODO: reuse one of tempValues1 or 2. will need to change keys from int
  // to unsigned int type.
  cudaMalloc(&deviceKSmallestKeys, numValues * sizeof(int));
  cudaMalloc(&tempValues1, numValues * sizeof(uint32_cu));
  cudaMalloc(&tempValues2, numValues * sizeof(uint32_cu));

  // allocate a host variable that is used to alter `k` after each iteration
  uint32_cu *toSubtract;
  toSubtract = (uint32_cu *)malloc(sizeof(uint32_cu));

  // declare values that are altered over the iterations
  uint32_cu kthSmallest = 0;
  int currNumValues = numValues;
  int currK = k;
  uint32_cu *currValues = values;
  uint32_cu *tempValues = tempValues1;

  // iterate over four 8-bit chunks in a 32-bit integer to find kth smallest
  // value
  for (int position = 1; position <= 4; ++position) {
    // Collect histogram. This is the most expensive part of the algorithm
    // and accounts for 90%+ of the duration. For this reason, we are putting
    // in the effort to make two implementations--one that uses shared memory
    // and one that doesn't--for different iterations.
    cudaMemset(histogram, 0, 256 * sizeof(int));
    if (position == 1) // TODO: Change to `numValues`-based threshold
      collectHistogramSharedMem<<<numBlocks, blockSize>>>(
          currNumValues, currValues, histogram, position);
    else
      collectHistogram<<<numBlocks, blockSize>>>(currNumValues, currValues,
                                                 histogram, position);
    cudaDeviceSynchronize();

    // compute prefix sums
    cudaMemset(prefixSums, 0, 256 * sizeof(int));
    thrust::inclusive_scan(thrust::device, histogram, histogram + 256,
                           prefixSums);
    // find pivot bin
    int *pivotPtr = thrust::lower_bound(thrust::device, prefixSums,
                                        prefixSums + 256, currK);
    uint32_cu pivot = (uint32_cu)(pivotPtr - prefixSums);

    // record pivot bits in the correct position in `kthSmallest`
    kthSmallest = kthSmallest | (pivot << ((sizeof(uint32_cu) - position) * 8));

    // copy integers from their corresponding pivot from `currValues` into
    // `temp` and record the count
    uint32_cu *copy_ifResult =
        thrust::copy_if(thrust::device, currValues, currValues + currNumValues,
                        tempValues, belongsToPivotBin(position, pivot));
    int count = (int)(copy_ifResult - tempValues);

    // in next iteration we change `currK` to account for all elements in lesser
    // bins, `currNumValues` to account for the elements only in the pivot bin,
    // and `currValues` to refer to the temporarily allocated memory
    currNumValues = count;
    if (pivot > 0) {
      cudaMemcpy(toSubtract, &prefixSums[pivot - 1], sizeof(uint32_cu),
                 cudaMemcpyDeviceToHost);
      currK -= *toSubtract;
    }

    // update `currValues` and cycle between temporary arrays
    if (currValues == values || currValues == tempValues2) {
      currValues = tempValues1;
      tempValues = tempValues2;
    } else if (currValues == tempValues1) {
      currValues = tempValues2;
      tempValues = tempValues1;
    }
  }

  // copy keys whose values are below threshold into `deviceKSmallestKeys`
  thrust::copy_if(thrust::device, keys, keys + numValues, values,
                  deviceKSmallestKeys,
                  valueBelowThreshold(values, kthSmallest));

  // Re-use tempValues1 to copy all values less than or equal to `kthSmallest`
  thrust::copy_if(thrust::device, values, values + numValues, tempValues1,
                  valueBelowThreshold(values, kthSmallest));

  // copy from `deviceKSmallestKeys` into host `kSmallestKeys` specified by
  // caller
  cudaMemcpy(kSmallestKeys, deviceKSmallestKeys, k * sizeof(int),
             cudaMemcpyDeviceToHost);

  // copy from `tempValues1` into host `kSmallestValues` specified by caller
  cudaMemcpy(kSmallestValues, tempValues1, k * sizeof(uint32_cu),
             cudaMemcpyDeviceToHost);

  cudaFree(histogram);
  cudaFree(prefixSums);
  cudaFree(tempValues1);
  cudaFree(tempValues2);
  cudaFree(deviceKSmallestKeys);

  free(toSubtract);

  return kthSmallest;
}
