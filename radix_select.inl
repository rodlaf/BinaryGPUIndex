#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

__device__ unsigned positionBits(unsigned value, int position) {
  return (value >> ((sizeof(unsigned) - position) * 8)) & 0xff;
}

__global__ void collectHistogram(int numValues, unsigned *values,
                                 unsigned *histogram, int position) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < numValues; i += stride) {
    unsigned bin = positionBits(values[i], position);
    atomicAdd(&histogram[bin], 1);
  }
}

/*
  This histogram collector uses shared memory. This is a performance improvement
  only for when numValues is very large (e.g., this is only used in the first
  iteration). TODO: combine two versions somehow
*/
__global__ void collectHistogramSharedMem(int numValues, unsigned *values,
                                          unsigned *histogram, int position) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int id = threadIdx.x;

  // first collect in per-block shared memory
  __shared__ unsigned sharedHistogram[256];

  if (id < 256) {
    sharedHistogram[id] = 0; // need to zero out first
  }

  __syncthreads();

  for (int i = idx; i < numValues; i += stride) {
    unsigned bin = positionBits(values[i], position);
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
  unsigned pivot;

  belongsToPivotBin(int position, unsigned pivot)
      : position(position), pivot(pivot) {}

  __device__ bool operator()(unsigned value) {
    return positionBits(value, position) == pivot;
  }
};
struct valueBelowThreshold {
  unsigned *values;
  unsigned threshold;

  valueBelowThreshold(unsigned *values, unsigned threshold)
      : values(values), threshold(threshold) {}

  __device__ bool operator()(unsigned value) { return value <= threshold; }
};

void radix_select(unsigned *values, unsigned *keys, int numValues, int k,
                       unsigned *kSmallestValues, unsigned *kSmallestKeys) {
  int blockSize = 512;
  int numBlocks = (numValues + blockSize - 1) / blockSize;

  // allocate histogram, prefix sum, and temporary arrays
  unsigned *histogram, *prefixSums;
  // TODO: cut down on size of these
  unsigned *tempValues1, *tempValues2;

  cudaMalloc(&histogram, 256 * sizeof(unsigned));
  cudaMalloc(&prefixSums, 256 * sizeof(unsigned));
  // TODO: experiment speed change when having these passed instead of 
  // allocated on the fly every time a call is made
  cudaMalloc(&tempValues1, numValues * sizeof(unsigned));
  cudaMalloc(&tempValues2, numValues * sizeof(unsigned));

  // allocate a host variable that is used to alter `k` after each iteration
  unsigned *toSubtract;
  toSubtract = (unsigned *)malloc(sizeof(unsigned));

  // declare values that are altered over the iterations
  unsigned kthSmallest = 0;
  int currNumValues = numValues;
  int currK = k;
  unsigned *currValues = values;
  unsigned *tempValues = tempValues1;

  // iterate over four 8-bit chunks in a 32-bit integer to find kth smallest
  // value
  for (int position = 1; position <= 4; ++position) {
    // Collect histogram. This is the most expensive part of the algorithm
    // and accounts for 90%+ of the duration. For this reason, we are putting
    // in the effort to make two implementations--one that uses shared memory
    // and one that doesn't--for different iterations.
    cudaMemset(histogram, 0, 256 * sizeof(unsigned));
    if (position == 1) // TODO: Change to `numValues`-based threshold
      collectHistogramSharedMem<<<numBlocks, blockSize>>>(
          currNumValues, currValues, histogram, position);
    else
      collectHistogram<<<numBlocks, blockSize>>>(currNumValues, currValues,
                                                 histogram, position);
    cudaDeviceSynchronize();

    // compute prefix sums
    cudaMemset(prefixSums, 0, 256 * sizeof(unsigned));
    thrust::inclusive_scan(thrust::device, histogram, histogram + 256,
                           prefixSums);
    // find pivot bin
    unsigned *pivotPtr = thrust::lower_bound(thrust::device, prefixSums,
                                        prefixSums + 256, currK);
    unsigned pivot = (unsigned)(pivotPtr - prefixSums);

    // record pivot bits in the correct position in `kthSmallest`
    kthSmallest = kthSmallest | (pivot << ((sizeof(unsigned) - position) * 8));

    // copy integers from their corresponding pivot from `currValues` into
    // `temp` and record the count
    unsigned *copy_ifResult =
        thrust::copy_if(thrust::device, currValues, currValues + currNumValues,
                        tempValues, belongsToPivotBin(position, pivot));
    int count = (int)(copy_ifResult - tempValues);

    // in next iteration make `currNumValues` the number of elements in the 
    // pivot bin and subtract from `currK` the number of elements in lesser 
    // bins.  
    currNumValues = count;
    if (pivot > 0) {
      cudaMemcpy(toSubtract, &prefixSums[pivot - 1], sizeof(unsigned),
                 cudaMemcpyDeviceToHost);
      currK -= *toSubtract;
    }

    // update `currValues` pointer and cycle between temporary arrays
    if (currValues == values || currValues == tempValues2) {
      currValues = tempValues1;
      tempValues = tempValues2;
    } else if (currValues == tempValues1) {
      currValues = tempValues2;
      tempValues = tempValues1;
    }
  }

  // reuse `tempValues1` to copy keys whose values are below threshold
  thrust::copy_if(thrust::device, keys, keys + numValues, values,
                  tempValues1,
                  valueBelowThreshold(values, kthSmallest));

  // reuse `tempValues2` to copy all values less than or equal to `kthSmallest`
  thrust::copy_if(thrust::device, values, values + numValues, tempValues2,
                  valueBelowThreshold(values, kthSmallest));

  // copy from `tempValues1` into host `kSmallestKeys` specified by
  // caller
  cudaMemcpy(kSmallestKeys, tempValues1, k * sizeof(unsigned),
             cudaMemcpyDeviceToHost);

  // copy from `tempValues2` into host `kSmallestValues` specified by caller
  cudaMemcpy(kSmallestValues, tempValues2, k * sizeof(unsigned),
             cudaMemcpyDeviceToHost);

  cudaFree(histogram);
  cudaFree(prefixSums);
  cudaFree(tempValues1);
  cudaFree(tempValues2);

  free(toSubtract);
}
