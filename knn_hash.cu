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

typedef unsigned long long int uint64_cu;

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
        // add address to index for different results at different addresses
        p[idx] = hash((uint64_cu)p + idx); 
        idx += blockDim.x * gridDim.x;
    }
}

__device__ void cosineDistance(uint64_cu *a, uint64_cu *b, float *dest) {
  // __popcll computes the Hamming Weight of an integer (e.g., number of bits
  // that are 1)
  float a_dot_b = (float)__popcll(*a & *b);
  float a_dot_a = (float)__popcll(*a);
  float b_dot_b = (float)__popcll(*b);

  *dest = 1 - (a_dot_b / (sqrt(a_dot_a) * sqrt(b_dot_b)));
}

__global__ void computeDistances(int numIndexes, uint64_cu *query,
                                 uint64_cu *indexes, float *distances) {
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
  int numIndexes = 970000000; // rough maximum on 24gb of GPU memory
  int k = 100;

  int blockSize = 256;
  int numBlocks = (numIndexes + blockSize - 1) / blockSize;

  // host memory
  uint64_cu *hostQuery;
  float *kNearestDistances;
  uint64_cu *kNearestIndexes;
  hostQuery = (uint64_cu *)malloc(sizeof(uint64_cu));
  kNearestDistances = (float *)malloc(k * sizeof(float));
  kNearestIndexes = (uint64_cu *)malloc(k * sizeof(uint64_cu));

  // device memory
  uint64_cu *query, *indexes;
  float *distances;
  cudaMalloc(&query, sizeof(uint64_cu));
  cudaMalloc(&indexes, numIndexes * sizeof(uint64_cu));
  cudaMalloc(&distances, numIndexes * sizeof(float));

  // generate random indexes on device
  randf<<<numBlocks, blockSize>>>(indexes, numIndexes);

  // generate random query on device
  randf<<<1, 1>>>(query, 1);
  cudaMemcpy(hostQuery, query, sizeof(uint64_t), cudaMemcpyDeviceToHost);

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

    // problem: needs a ton of space, contributes most to duration; no easy
    // parallelizable way to get k smallest values in an unsorted list of floats
    // ~11gb allocated on GPU by this point, needs more than double to execute 
    thrust::sort_by_key(thrust::device, distances, distances + numIndexes,
                        indexes);

    // copy k nearest distances and indexes from device to host
    cudaMemcpy(kNearestDistances, distances, k * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(kNearestIndexes, indexes, k * sizeof(uint64_cu),
               cudaMemcpyDeviceToHost);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  printf("Execution time:  %.3f ms \n", time);

  // print results
  printf("Query: ");
  printBits(hostQuery);
  for (int i = 0; i < k; ++i) {
    printf("%5d: %8.8f ", i + 1, kNearestDistances[i]);
    printBits(&kNearestIndexes[i]);
  }

  // free device memory
  cudaFree(query);
  cudaFree(indexes);
  cudaFree(distances);

  // free host memory
  free(hostQuery);
  free(kNearestDistances);
  free(kNearestIndexes);

  return 0;
}