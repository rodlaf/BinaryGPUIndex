#include <bitset>
#include <iostream>
#include <math.h>
#include <sys/time.h>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include "kNearestNeighbors.h"

// murmur32 hash function
__device__ unsigned hash(unsigned a) {
  a = (a ^ 61) ^ (a >> 16);
  a = a + (a << 3);
  a = a ^ (a >> 4);
  a = a * 0x27d4eb2d;
  a = a ^ (a >> 15);
  return a;
}

template <typename T> __global__ void rand(T *vectors, int n, size_t D) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < n; i += stride) {
    for (int d = 0; d < D; ++d) {
      vectors[i * D + d] = (float)hash(i * D + d) / (float)UINT_MAX;
    }
  }
}

__host__ void printBits(uint64_cu *x) {
  std::bitset<sizeof(uint64_cu) * CHAR_BIT> b(*x);
  std::cout << b << std::endl;
}

template <typename T>
void retrieveVectorsFromKeys(T *vectors, size_t D, unsigned *keys, int numKeys,
                             T *retrieved) {
  for (int i = 0; i < numKeys; ++i)
    cudaMemcpy(retrieved + i * D, vectors + keys[i] * D, D * sizeof(T),
               cudaMemcpyDeviceToHost);
}

int main(void) {
  typedef float T;

  int numIndexes = 10000000;
  const size_t D = 2;
  int K = 5;

  int blockSize = 1024;
  int numBlocks = (numIndexes + blockSize - 1) / blockSize;

  // allocate K nearest distances, keys, and vectors
  float *kNearestDistances;
  unsigned *kNearestKeys;
  T *kNearestVectors;
  cudaMallocManaged(&kNearestDistances, K * sizeof(unsigned));
  cudaMallocManaged(&kNearestKeys, K * sizeof(unsigned));
  cudaMallocManaged(&kNearestVectors, K * D * sizeof(T));

  // allocate space on device for query and vectors
  T *query;
  T *vectors;
  cudaMallocManaged(&query, D * sizeof(T));
  cudaMalloc(&vectors, numIndexes * D * sizeof(T));

  // allocate and initalize keys on device
  unsigned *keys;
  cudaMalloc(&keys, numIndexes * sizeof(unsigned));
  thrust::sequence(thrust::device, keys, keys + numIndexes);

  // allocate working memory on device
  unsigned *workingMem1, *workingMem2, *workingMem3;
  cudaMalloc(&workingMem1, numIndexes * sizeof(unsigned));
  cudaMalloc(&workingMem2, numIndexes * sizeof(unsigned));
  cudaMalloc(&workingMem3, numIndexes * sizeof(unsigned));

  // generate random vectors on device
  rand<T><<<numBlocks, blockSize>>>(vectors, numIndexes, D);
  cudaDeviceSynchronize();

  // generate random query on device and transfer to host
  rand<T><<<1, 1>>>(query, 1, D);
  cudaDeviceSynchronize();

  // run and time kNearestNeighbors call
  float time;
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  kNearestNeighbors<T>(vectors, D, keys, query, numIndexes, K,
                       kNearestDistances, kNearestKeys, workingMem1,
                       workingMem2, workingMem3);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  printf("Execution time:  %.3f ms \n\n", time);

  // retrieve vectors from relevant keys
  retrieveVectorsFromKeys<T>(vectors, D, kNearestKeys, K, kNearestVectors);

  // print results
  printf("Query:\n");
  for (int d = 0; d < D; ++d) {
    printf("%f ", query[d]);
  }
  printf("\n\n");

  for (int i = 0; i < K; ++i) {
    printf("%d: %f \n", i, kNearestDistances[i]);
    for (int d = 0; d < D; ++d) {
      printf("%f ", kNearestVectors[i * D + d]);
    }
    printf("\n\n");
  }

  // free device memory
  cudaFree(query);
  cudaFree(vectors);
  cudaFree(keys);
  cudaFree(kNearestDistances);
  cudaFree(kNearestKeys);
  cudaFree(kNearestVectors);
  cudaFree(workingMem1);
  cudaFree(workingMem2);
  cudaFree(workingMem3);

  return 0;
}
