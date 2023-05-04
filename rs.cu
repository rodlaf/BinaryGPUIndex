#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "radix_select.cuh"

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
  int k = 20000;

  int blockSize = 512;
  int numBlocks = (n + blockSize - 1) / blockSize;

  // generate random numbers
  uint32_cu *xs;
  cudaMalloc(&xs, n * sizeof(uint32_cu));
  rand<<<numBlocks, blockSize>>>(n, xs);
  cudaDeviceSynchronize();

  // run radix select
  float time;
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  uint32_cu result = radix_select(xs, n, k);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  printf("Execution time:  %.3f ms \n", time);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  thrust::sort(thrust::device, xs, xs + n);
  uint32_cu *result2 = (uint32_cu *)malloc(sizeof(uint32_cu)); 
  cudaMemcpy(result2, &xs[k - 1], sizeof(uint32_cu), cudaMemcpyDeviceToHost);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  printf("Execution time:  %.3f ms \n", time);

  printf("Result: %u\n", result);
  printf("Result2: %u\n", *result2);

  cudaFree(xs);

  return 0;
}