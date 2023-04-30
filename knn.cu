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

__device__ void cosineSimilarity(uint64_cu *a, uint64_cu *b, float *dest) {
  // __popcll computes the Hamming Weight of an integer (e.g., number of bits
  // that are 1)
  float a_dot_b = (float)__popcll(*a & *b);
  float a_dot_a = (float)__popcll(*a);
  float b_dot_b = (float)__popcll(*b);

  *dest = a_dot_b / (sqrt(a_dot_a) * sqrt(b_dot_b));
}

__device__ void jaccardSimilarity(uint64_cu *a, uint64_cu *b, float *dest) {
  float intersectionBits = (float)__popcll(*a & *b);
  float unionBits = (float)__popcll(*a | *b);

  *dest = intersectionBits / unionBits;
}

__global__ void computeDistances(int numIndexes, uint64_cu *query,
                                 uint64_cu *indexes, float *distances) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < numIndexes; i += stride)
    cosineSimilarity(query, &indexes[i], &distances[i]);
}

__host__ void printBits(uint64_cu *x) {
  std::bitset<sizeof(uint64_cu) * CHAR_BIT> b(*x);
  std::cout << b << std::endl;
}

__host__ void sortByDistance(int numIndexes, int k, uint64_cu *query,
                             uint64_cu *indexes, float *distances, int *keys) {
  int blockSize = 256;
  int numBlocks = (numIndexes + blockSize - 1) / blockSize;

  computeDistances<<<numBlocks, blockSize>>>(numIndexes, query, indexes,
                                             distances);

  thrust::sequence(thrust::device, keys, keys + numIndexes);

  thrust::sort_by_key(thrust::device, distances, distances + numIndexes, keys);
}

int main(void) {
  int numIndexes = 256;
  int k = 10;

  // host memory
  uint64_cu *hostQuery;
  uint64_cu *hostIndexes;
  int *kNearestKeys;
  float *kNearestDistances;
  uint64_cu *kNearestIndexes;
  hostQuery = (uint64_cu *)malloc(sizeof(uint64_cu));
  hostIndexes = (uint64_cu *)malloc(numIndexes * sizeof(uint64_cu));
  kNearestKeys = (int *)malloc(k * sizeof(int));
  kNearestDistances = (float *)malloc(k * sizeof(float));
  kNearestIndexes = (uint64_cu *)malloc(k * sizeof(uint64_cu));

  // device memory
  uint64_cu *query, *indexes;
  float *distances;
  int *keys;
  cudaMalloc(&query, sizeof(uint64_cu));
  cudaMalloc(&indexes, numIndexes * sizeof(uint64_cu));
  cudaMalloc(&distances, numIndexes * sizeof(float));
  cudaMalloc(&keys, numIndexes * sizeof(int));

  // generate indexes on host and transfer to device
  thrust::default_random_engine rng(1234);
  thrust::uniform_int_distribution<uint64_cu> uniDist(0, UINT64_MAX);
  thrust::generate(hostIndexes, hostIndexes + numIndexes,
                   [&] { return uniDist(rng); });
  cudaMemcpy(indexes, hostIndexes, numIndexes * sizeof(uint64_cu),
             cudaMemcpyHostToDevice);
  free(hostIndexes);

  // generate query on host and transfer to device
  *hostQuery = uniDist(rng);
  cudaMemcpy(query, hostQuery, sizeof(uint64_cu), cudaMemcpyHostToDevice);
  // free(hostQuery);

  float time;
  cudaEvent_t start, stop;

  cudaDeviceSynchronize();

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  int blockSize = 256;
  int numBlocks = (numIndexes + blockSize - 1) / blockSize;

  computeDistances<<<numBlocks, blockSize>>>(numIndexes, query, indexes,
                                             distances);

  thrust::sequence(thrust::device, keys, keys + numIndexes);

  thrust::sort_by_key(thrust::device, distances, distances + numIndexes, keys);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  printf("Execution time:  %.3f ms \n", time);

  cudaMemcpy(kNearestKeys, keys, k * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(kNearestDistances, distances, k * sizeof(float), cudaMemcpyDeviceToHost);

  // for (int i = 0; i < k; ++i) {
  //   int idx = kNearestKeys[i];
  //   cudaMemcpy(&kNearestIndexes[idx], &indexes[idx], sizeof(uint64_cu),
  //              cudaMemcpyDeviceToHost);
  // }

  for (int i = 0; i < k; ++i) {
    printf("%d: %8d  %8.8f\n", i, kNearestKeys[i], kNearestDistances[i]);
    // printBits(&kNearestIndexes[i]);
  }

  cudaFree(query);
  cudaFree(indexes);
  cudaFree(distances);

  return 0;
}