#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "radix_select.h"

__device__ unsigned hash(unsigned a) {
  a = (a ^ 61) ^ (a >> 16);
  a = a + (a << 3);
  a = a ^ (a >> 4);
  a = a * 0x27d4eb2d;
  a = a ^ (a >> 15);
  return a;
}

__global__ void rand(int n, unsigned *xs) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < n; i += stride)
    xs[idx] = hash(~idx);
}

int main() {
  int n = 1 << 30;
  int k = 10;

  int blockSize = 512;
  int numBlocks = (n + blockSize - 1) / blockSize;

  // generate random numbers
  unsigned *xs;
  cudaMalloc(&xs, n * sizeof(unsigned));
  rand<<<numBlocks, blockSize>>>(n, xs);
  cudaDeviceSynchronize();

  // allocate and initalize keys on device
  unsigned *keys;
  cudaMalloc(&keys, n * sizeof(unsigned));
  thrust::sequence(thrust::device, keys, keys + n);

  // allocate kSmallestKeys and kSmallestValues on host
  unsigned *kSmallestKeys = (unsigned *)malloc(k * sizeof(unsigned));
  unsigned *kSmallestValues = (unsigned *)malloc(k * sizeof(unsigned));

  unsigned *tempValues1, *tempValues2;
  cudaMalloc(&tempValues1, n * sizeof(unsigned));
  cudaMalloc(&tempValues2, n * sizeof(unsigned));


  // run radix select
  float time;
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  radix_select(xs, keys, n, k, kSmallestValues, kSmallestKeys, tempValues1, tempValues2);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  printf("Execution time:  %.3f ms \n", time);

  for (int i = 0; i < k; ++i) {
    printf("kSmallestKeys: %d: %u\n", i, kSmallestKeys[i]);
  }
  for (int i = 0; i < k; ++i) {
    printf("kSmallestValues: %d: %u\n", i, kSmallestValues[i]);
  }

  // // run thrust sort
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start, 0);

  // thrust::sort(thrust::device, xs, xs + n);

  // cudaEventRecord(stop, 0);
  // cudaEventSynchronize(stop);
  // cudaEventElapsedTime(&time, start, stop);

  // printf("Execution time:  %.3f ms \n", time);

  cudaFree(xs);
  cudaFree(keys);
  cudaFree(tempValues1);
  cudaFree(tempValues2);

  free(kSmallestKeys);

  return 0;
}