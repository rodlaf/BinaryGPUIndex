#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "radix_select.h"

__device__ uint32_cu hash(uint32_cu a) {
  a = (a ^ 61) ^ (a >> 16);
  a = a + (a << 3);
  a = a ^ (a >> 4);
  a = a * 0x27d4eb2d;
  a = a ^ (a >> 15);
  return a;
}

__global__ void rand(int n, uint32_cu *xs) {
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
  uint32_cu *xs;
  cudaMalloc(&xs, n * sizeof(uint32_cu));
  rand<<<numBlocks, blockSize>>>(n, xs);
  cudaDeviceSynchronize();

  // allocate and initalize keys on device
  int *keys;
  cudaMalloc(&keys, n * sizeof(int));
  thrust::sequence(thrust::device, keys, keys + n);

  // allocate kSmallestKeys and kSmallestValues on host
  int *kSmallestKeys = (int *)malloc(k * sizeof(int));
  uint32_cu *kSmallestValues = (uint32_cu *)malloc(k * sizeof(int));

  // run radix select
  float time;
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  radix_select(xs, keys, n, k, kSmallestValues, kSmallestKeys);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  printf("Execution time:  %.3f ms \n", time);

  for (int i = 0; i < k; ++i) {
    printf("kSmallestKeys: %d: %d\n", i, kSmallestKeys[i]);
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

  free(kSmallestKeys);

  return 0;
}