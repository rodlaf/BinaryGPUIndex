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
    p[idx] = hash((uint64_cu)~idx);
    idx += blockDim.x * gridDim.x;
  }
}

__host__ void printBits(uint64_cu *x) {
  std::bitset<sizeof(uint64_cu) * CHAR_BIT> b(*x);
  std::cout << b << std::endl;
}

__global__ void retrieveVectorsFromKeys(uint64_cu *vectors, unsigned *keys,
                                        int numKeys, uint64_cu *retrieved) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < numKeys; i += stride)
    retrieved[i] = vectors[keys[i]];
}

int main(void) {
  int numIndexes = 970000000;
  int k = 10;

  int blockSize = 1024;
  int numBlocks = (numIndexes + blockSize - 1) / blockSize;

  // allocate space on host for query, and k nearest indexes
  uint64_cu *hostQuery;
  hostQuery = (uint64_cu *)malloc(sizeof(uint64_cu));

  // allocate k nearest distances, keys, and indexes on device
  float *kNearestDistances;
  unsigned *kNearestKeys;
  uint64_cu *kNearestIndexes;
  cudaMalloc(&kNearestDistances, k * sizeof(unsigned));
  cudaMalloc(&kNearestKeys, k * sizeof(unsigned));
  cudaMalloc(&kNearestIndexes, k * sizeof(uint64_cu));

  // allocate host versions of kNearestDistances, kNearestKeys, and
  // kNearestIndexes
  float *hostKNearestDistances;
  unsigned *hostKNearestKeys;
  uint64_cu *hostKNearestIndexes;
  hostKNearestDistances = (float *)malloc(k * sizeof(float));
  hostKNearestKeys = (unsigned *)malloc(k * sizeof(unsigned));
  hostKNearestIndexes = (uint64_cu *)malloc(k * sizeof(uint64_cu));

  // allocate space on device for query and indexes
  uint64_cu *query, *indexes;
  cudaMalloc(&query, sizeof(uint64_cu));
  cudaMalloc(&indexes, numIndexes * sizeof(uint64_cu));

  // allocate and initalize keys on device
  // TODO: would make code cleaner but would add ~6ms on ~1B vectors to
  // move initalization of keys into kNearestNeighbors (thrust::sequence)
  // such that the function does not use "keys" terminology and instead
  // is defined as returning the array indexes of the closest k vectors.
  unsigned *keys;
  cudaMalloc(&keys, numIndexes * sizeof(unsigned));
  thrust::sequence(thrust::device, keys, keys + numIndexes);

  // allocate working memory on device
  unsigned *workingMem1, *workingMem2, *workingMem3;
  cudaMalloc(&workingMem1, numIndexes * sizeof(unsigned));
  cudaMalloc(&workingMem2, numIndexes * sizeof(unsigned));
  cudaMalloc(&workingMem3, numIndexes * sizeof(unsigned));

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
                    kNearestKeys, workingMem1, workingMem2, workingMem3);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  printf("Execution time:  %.3f ms \n", time);

  // retrieve vectors from relevant keys
  retrieveVectorsFromKeys<<<1, blockSize>>>(indexes, kNearestKeys, k,
                                            kNearestIndexes);
  cudaDeviceSynchronize();

  // copy results from device to host
  cudaMemcpy(hostKNearestDistances, kNearestDistances, k * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(hostKNearestKeys, kNearestKeys, k * sizeof(unsigned),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(hostKNearestIndexes, kNearestIndexes, k * sizeof(uint64_cu),
             cudaMemcpyDeviceToHost);

  // print results
  printf("Query: ");
  printBits(hostQuery);
  for (int i = 0; i < k; ++i) {
    printf("%d: %f ", i, hostKNearestDistances[i]);
    printBits(&hostKNearestIndexes[i]);
  }

  // free device memory
  cudaFree(query);
  cudaFree(indexes);
  cudaFree(keys);
  cudaFree(kNearestDistances);
  cudaFree(kNearestKeys);
  cudaFree(kNearestIndexes);
  cudaFree(workingMem1);
  cudaFree(workingMem2);
  cudaFree(workingMem3);

  // free host memory
  free(hostQuery);
  free(hostKNearestDistances);
  free(hostKNearestKeys);

  return 0;
}