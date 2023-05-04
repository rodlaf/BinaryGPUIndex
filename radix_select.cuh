#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

// 32 bit unsigned integer
typedef unsigned int uint32_cu;

__device__ uint32_cu positionBits(uint32_cu x, int position) {
  return (x >> ((sizeof(uint32_cu) - position) * 8)) & 0xff;
}

__global__ void collectHistogram(int n, uint32_cu *xs, int *histogram,
                                 int position) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < n; i += stride) {
    uint32_cu bin = positionBits(xs[i], position);
    atomicAdd(&histogram[bin], 1);
  }
}

// used in thrust::copy_if as a predicate
struct belongsToPivotBin {
  int position;
  uint32_cu pivot;

  belongsToPivotBin(int position, uint32_cu pivot)
      : position(position), pivot(pivot) {}

  __device__ bool operator()(const uint32_cu x) {
    return positionBits(x, position) == pivot;
  }
};

uint32_cu radix_select(uint32_cu *xs, int n, int k) {
  int blockSize = 512;
  int numBlocks = (n + blockSize - 1) / blockSize;

  // allocate histogram, prefix sum, and temporary arrays
  int *histogram, *prefixSums;
  uint32_cu *temp;

  cudaMalloc(&histogram, 256 * sizeof(int));
  cudaMallocManaged(&prefixSums, 256 * sizeof(int));
  cudaMalloc(&temp, n * sizeof(uint32_cu));

  // result
  uint32_cu result = 0;

  // iterate over four 8-bit chunks in a 32-bit integer
  for (int position = 1; position <= 4; ++position) {
    // collect histogram
    cudaMemset(histogram, 0, 256 * sizeof(int));
    collectHistogram<<<numBlocks, blockSize>>>(n, xs, histogram, position);
    cudaDeviceSynchronize();

    // compute prefix sums
    cudaMemset(prefixSums, 0, 256 * sizeof(int));
    thrust::inclusive_scan(thrust::device, histogram, histogram + 256,
                           prefixSums);
    // find pivot bin
    int *pivotPtr =
        thrust::lower_bound(thrust::device, prefixSums, prefixSums + 256, k);
    uint32_cu pivot = (uint32_cu)(pivotPtr - prefixSums);

    // record in pivot bin in result
    result = result | (pivot << ((sizeof(uint32_cu) - position) * 8));

    // copy integers from their corresponding pivot from `xs` into `temp` and 
    // record the count
    uint32_cu *copy_ifResult = thrust::copy_if(thrust::device, xs, xs + n, temp,
                                     belongsToPivotBin(position, pivot));
    int count = (int)(copy_ifResult - temp);

    // in next iteration we change k to account for all elements in lesser
    // bins, n to account for the elements only in the pivot bin, and xs 
    // to refer to the temporarily allocated memory
    n = count;
    if (pivot > 0)
      k -= prefixSums[pivot - 1];
    xs = temp; // this will only make a diference in the first iteration
  }

  cudaFree(histogram);
  cudaFree(prefixSums);
  cudaFree(temp);

  return result;
}
