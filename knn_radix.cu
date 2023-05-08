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

#include "knn.inl"

// murmur64 hash function
__device__ uint64_cu hash(uint64_cu h) {
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53L;
  h ^= h >> 33;
  return h;
}

__global__ void randf(uint64_cu *p, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < n) {
    // hash address
    p[idx] = hash((uint64_cu)&p[idx]);
    idx += blockDim.x * gridDim.x;
  }
}

__host__ void printBits(uint64_cu *x) {
  std::bitset<sizeof(uint64_cu) * CHAR_BIT> b(*x);
  std::cout << b << std::endl;
}

int main(void) {
  int numIndexes = 950000000;
  int k = 10;

  int blockSize = 256;
  int numBlocks = (numIndexes + blockSize - 1) / blockSize;

  // allocate space on host for query, k nearest distances, and k nearest
  // indexes
  uint64_cu *hostQuery;
  float *kNearestDistances;
  uint64_cu *kNearestIndexes;
  hostQuery = (uint64_cu *)malloc(sizeof(uint64_cu));
  kNearestDistances = (float *)malloc(k * sizeof(float));
  kNearestIndexes = (uint64_cu *)malloc(k * sizeof(uint64_cu));

  // allocate space to receive k nearest keys on host
  unsigned *kNearestKeys;
  kNearestKeys = (unsigned *)malloc(k * sizeof(unsigned));

  // allocate space on device for query and indexes
  uint64_cu *query, *indexes;
  cudaMalloc(&query, sizeof(uint64_cu));
  cudaMalloc(&indexes, numIndexes * sizeof(uint64_cu));

  unsigned *distances;
  cudaMalloc(&distances, numIndexes * sizeof(unsigned));

  // allocate and initalize keys on device
  unsigned *keys;
  cudaMalloc(&keys, numIndexes * sizeof(unsigned));
  thrust::sequence(thrust::device, keys, keys + numIndexes);

  // allocate working memory
  unsigned *workingMem1, *workingMem2;
  cudaMalloc(&workingMem1, numIndexes * sizeof(unsigned));
  cudaMalloc(&workingMem2, numIndexes * sizeof(unsigned));

  // generate random indexes on device
  randf<<<numBlocks, blockSize>>>(indexes, numIndexes);
  cudaDeviceSynchronize();

  // generate random query on device and transfer to host
  randf<<<1, 1>>>(query, 1);
  cudaDeviceSynchronize();
  cudaMemcpy(hostQuery, query, sizeof(uint64_t), cudaMemcpyDeviceToHost);

  // run and time kNearestNeighbors call
  float time;
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  kNearestNeighbors(indexes, keys, query, numIndexes, k, kNearestDistances,
                    kNearestIndexes, kNearestKeys, distances, workingMem1,
                    workingMem2);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  printf("Execution time:  %.3f ms \n", time);

  // print results
  printf("Query: ");
  printBits(hostQuery);
  for (int i = 0; i < k; ++i) {
    printf("%d: %f ", i, kNearestDistances[i]);
    printBits(&kNearestIndexes[i]);
  }

  // free device memory
  cudaFree(query);
  cudaFree(indexes);
  cudaFree(distances);
  cudaFree(keys);
  cudaFree(workingMem1);
  cudaFree(workingMem2);

  // free host memory
  free(hostQuery);
  free(kNearestDistances);
  free(kNearestIndexes);
  free(kNearestKeys);

  return 0;
}