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

#include "radix_select.h"

typedef unsigned long long int uint64_cu;

// murmur64 hash function
__device__ uint64_cu hash(uint64_cu h) {
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53L;
  h ^= h >> 33;
  return h;
}

__global__ void randf(uint64_cu *p, int n){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while(idx < n){
        // hash address
        p[idx] = hash((uint64_cu)&p[idx]); 
        idx += blockDim.x * gridDim.x;
    }
}

// TEMPORARILY MODIFIED to return an unsigned integer
__device__ void cosineDistance(uint64_cu *a, uint64_cu *b, unsigned int *dest) {
  // __popcll computes the Hamming Weight of an integer (e.g., number of bits
  // that are 1)
  float a_dot_b = (float)__popcll(*a & *b);
  float a_dot_a = (float)__popcll(*a);
  float b_dot_b = (float)__popcll(*b);

  *dest = (unsigned int)((1 - (a_dot_b / (sqrt(a_dot_a) * sqrt(b_dot_b)))) * UINT32_MAX);
}

__global__ void computeDistances(int numIndexes, uint64_cu *query,
                                 uint64_cu *indexes, unsigned int *distances) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < numIndexes; i += stride)
    cosineDistance(query, &indexes[i], &distances[i]);
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

  // allocate host memory
  uint64_cu *hostQuery;
  unsigned int *kNearestDistances;
  uint64_cu *kNearestIndexes;
  int *kNearestKeys;
  hostQuery = (uint64_cu *)malloc(sizeof(uint64_cu));
  kNearestDistances = (unsigned int *)malloc(k * sizeof(unsigned int));
  kNearestIndexes = (uint64_cu *)malloc(k * sizeof(uint64_cu));
  kNearestKeys = (int *)malloc(k * sizeof(int));


  // allocate device memory
  uint64_cu *query, *indexes;
  unsigned int *distances;
  cudaMalloc(&query, sizeof(uint64_cu));
  cudaMalloc(&indexes, numIndexes * sizeof(uint64_cu));
  cudaMalloc(&distances, numIndexes * sizeof(unsigned int));

  // generate random indexes on device
  randf<<<numBlocks, blockSize>>>(indexes, numIndexes);
  cudaDeviceSynchronize();

  // generate random query on device and transfer to host
  randf<<<1, 1>>>(query, 1);
  cudaDeviceSynchronize();
  cudaMemcpy(hostQuery, query, sizeof(uint64_t), cudaMemcpyDeviceToHost);

  // allocate and initalize keys on device
  int *keys;
  cudaMalloc(&keys, numIndexes * sizeof(int));
  thrust::sequence(thrust::device, keys, keys + numIndexes);

  float time;
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // compute and retrieve k nearest neighbors of a query index
  {
    // ~23ms on ~1B indexes
    computeDistances<<<numBlocks, blockSize>>>(numIndexes, query, indexes,
                                               distances);

    radix_select(distances, keys, numIndexes, k, kNearestDistances, kNearestKeys);

    // copy indicated indexes from device to host
    // TODO: should use a kernel for this
    for (int i = 0; i < k; ++i) {
      int idx = kNearestKeys[i];
      cudaMemcpy(&kNearestIndexes[i], &indexes[idx], sizeof(uint64_cu),
                 cudaMemcpyDeviceToHost);
    }
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  printf("Execution time:  %.3f ms \n", time);

  // print results
  printf("Query: ");
  printBits(hostQuery);
  for (int i = 0; i < k; ++i) {
    printf("%d: %u ", i, kNearestDistances[i]);
    printBits(&kNearestIndexes[i]);
  }

  // free device memory
  cudaFree(query);
  cudaFree(indexes);
  cudaFree(distances);
  cudaFree(keys);

  // free host memory
  free(hostQuery);
  free(kNearestDistances);
  free(kNearestIndexes);
  free(kNearestKeys);

  return 0;
}