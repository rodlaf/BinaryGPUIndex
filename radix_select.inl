#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>

/*
  E.g., if value is 100000 10111111 0000001 11110111 then
  positionBits(value, 1) == 100000, positionBits(value, 2) is 10111111,
  positionBits(value, 2) == 000001, and positionBits(value, 4) is 11110111.
*/
__device__ unsigned positionBits(unsigned value, int position) {
  return (value >> ((sizeof(unsigned) - position) * 8)) & 0xff;
}

/*
  Collect histogram.
*/
__global__ void collectHistogram(int numValues, unsigned *values,
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

  __device__ bool operator()(unsigned value) { return value < threshold; }
};
struct valueEqualToThreshold {
  unsigned *values;
  unsigned threshold;

  valueEqualToThreshold(unsigned *values, unsigned threshold)
      : values(values), threshold(threshold) {}

  __device__ bool operator()(unsigned value) { return value == threshold; }
};

void radixSelect(unsigned *values, unsigned *keys, int numValues, int k,
                 unsigned *kSmallestValues, unsigned *kSmallestKeys,
                 unsigned *workingMem1, unsigned *workingMem2) {
  // allocate histogram, prefix sum, and temporary arrays
  unsigned *histogram, *prefixSums;

  cudaMalloc(&histogram, 256 * sizeof(unsigned));
  cudaMalloc(&prefixSums, 256 * sizeof(unsigned));

  // allocate a host variable that is used to alter `k` after each iteration
  unsigned *toSubtract;
  toSubtract = (unsigned *)malloc(sizeof(unsigned));

  // declare values that are altered over the iterations
  unsigned kthSmallestValue = 0;
  int currNumValues = numValues;
  int currK = k;
  unsigned *currValues = values;
  unsigned *tempValues = workingMem1;

  // iterate over four 8-bit chunks in a 32-bit integer to find kth smallest
  // value
  for (int position = 1; position <= 4; ++position) {
    int blockSize = 1024;
    int numBlocks = (currNumValues + blockSize - 1) / blockSize;

    // Collect histogram
    cudaMemset(histogram, 0, 256 * sizeof(unsigned));

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

    // record pivot bits in their corresponding position in `kthSmallestValue`
    kthSmallestValue =
        kthSmallestValue | (pivot << ((sizeof(unsigned) - position) * 8));

    if (position <= 3) {
      // copy integers from their corresponding pivot from `currValues` into
      // `temp` and record the count
      unsigned *copy_ifResult = thrust::copy_if(
          thrust::device, currValues, currValues + currNumValues, tempValues,
          belongsToPivotBin(position, pivot));
      unsigned binCount = copy_ifResult - tempValues;

      // in next iteration make `currNumValues` the number of elements in the
      // pivot bin and subtract from `currK` the number of elements in lesser
      // bins.
      currNumValues = binCount;
      if (pivot > 0) {
        cudaMemcpy(toSubtract, &prefixSums[pivot - 1], sizeof(unsigned),
                   cudaMemcpyDeviceToHost);
        currK -= *toSubtract;
      }

      // update `currValues` pointer and cycle between temporary arrays
      if (currValues == values || currValues == workingMem2) {
        currValues = workingMem1;
        tempValues = workingMem2;
      } else if (currValues == workingMem1) {
        currValues = workingMem2;
        tempValues = workingMem1;
      }
    }
  }

  // reuse `workingMem1` to copy keys whose values are strictly less than
  // `kthSmallestValue`
  unsigned *copy_ifResult = thrust::copy_if(
      thrust::device, keys, keys + numValues, values, workingMem1,
      valueBelowThreshold(values, kthSmallestValue));
  unsigned countLessThan = copy_ifResult - workingMem1;

  // copy keys whose values are equal to `kthSmallestValue` into the remaining
  // space in `workingMem1`.
  thrust::copy_if(thrust::device, keys, keys + numValues, values,
                  workingMem1 + countLessThan,
                  valueEqualToThreshold(values, kthSmallestValue));

  // reuse `workingMem2` to copy all values strictly less than
  // `kthSmallestValue`
  thrust::copy_if(thrust::device, values, values + numValues, workingMem2,
                  valueBelowThreshold(values, kthSmallestValue));

  // append onto values just copied into `workingMem1` values equal to
  // `kthSmallestValue` such that we have accounted for `k` total values
  thrust::fill(thrust::device, workingMem2 + countLessThan, workingMem2 + k,
               kthSmallestValue);

  // copy from `workingMem1` into host `kSmallestKeys` specified by
  // caller
  cudaMemcpy(kSmallestKeys, workingMem1, k * sizeof(unsigned),
             cudaMemcpyDeviceToHost);

  // copy from `workingMem2` into host `kSmallestValues` specified by caller
  cudaMemcpy(kSmallestValues, workingMem2, k * sizeof(unsigned),
             cudaMemcpyDeviceToHost);

  cudaFree(histogram);
  cudaFree(prefixSums);

  free(toSubtract);
}
