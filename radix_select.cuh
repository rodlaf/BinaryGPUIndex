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

void printBits(uint32_cu *x) {
  std::bitset<sizeof(uint32_cu) * CHAR_BIT> b(*x);
  std::cout << b << std::endl;
}


uint32_cu radix_select(uint32_cu *xs, int n, int k) {
  int blockSize = 512;
  int numBlocks = (n + blockSize - 1) / blockSize;

  // allocate histogram, prefix sum, and temporary arrays
  int *histogram, *prefixSums;
  uint32_cu *temp;

  cudaMallocManaged(&histogram, 256 * sizeof(int));
  cudaMallocManaged(&prefixSums, 256 * sizeof(int));
  cudaMallocManaged(&temp, n * sizeof(uint32_cu));

  // result
  uint32_cu result = 0;


  // iterate over four 8 bit chunks in a 32 bit integer
  for (int position = 1; position <= 4; ++position) {
    printf("\nn: %d\nk: %d\n", n, k);

    // for (int i = 0; i < n; i++) {
    //   printf("xs: %d: %u\n", i, xs[i]);
    // }

    // collect histogram
    cudaMemset(histogram, 0, 256 * sizeof(int));

    collectHistogram<<<numBlocks, blockSize>>>(n, xs, histogram, position);
    cudaDeviceSynchronize();

    // compute prefix sums
    cudaMemset(prefixSums, 0, 256 * sizeof(int));
    thrust::inclusive_scan(thrust::device, histogram, histogram + 256,
                           prefixSums);

    // for (int i = 0; i < 256; ++i) {
    //   printf("%d: %d\n", i, prefixSums[i]);
    // }


    // find pivot bin
    int *pivotPtr =
        thrust::lower_bound(thrust::device, prefixSums, prefixSums + 256, k);
    uint32_cu pivot = (uint32_cu)(pivotPtr - prefixSums);

    printf("pivot: %d\n", (int)pivot);
    printBits(&pivot);

    // record in pivot bin in result
    result = result | (pivot << ((sizeof(uint32_cu) - position) * 8));

    // copy integers from their corresponding pivot from xs into temp and 
    // record the count
    uint32_cu *copy_ifResult = thrust::copy_if(thrust::device, xs, xs + n, temp,
                                     belongsToPivotBin(position, pivot));
    int count = (int)(copy_ifResult - temp);

    printf("count: %d\n", count);

    // for (int i = 0; i < count; ++i) {
    //   printf("array: %d: %u\n", i, temp[i]);
    //   printBits(&temp[i]);
    // }


    // in next iteration, look only at `count` number of elements, in `temp`,
    // and we want to find the `n - count`th smallest element
    int toSubtractFromK = pivot == 0 ? 0 : prefixSums[pivot - 1];
    k -= toSubtractFromK;
    n = count;
    xs = temp; // this will only make a diference in the first iteration
  }

  cudaFree(histogram);
  cudaFree(prefixSums);
  cudaFree(temp);

  return result;
}
