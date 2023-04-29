#include <iostream>
#include <math.h>
#include <sys/time.h>
#include <bitset>

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

int main(void) {
  int numIndexes = 100;

  thrust::default_random_engine rng(1234);
  thrust::uniform_int_distribution<uint64_cu> dist(0, UINT64_MAX);

  uint64_cu *query, *indexes;
  float *distances;
  int *keys;

  cudaMallocManaged(&query, sizeof(uint64_cu));
  cudaMallocManaged(&indexes, numIndexes * sizeof(uint64_cu));
  cudaMallocManaged(&distances, numIndexes * sizeof(float));
  cudaMallocManaged(&keys, numIndexes * sizeof(int));

  thrust::generate(indexes, indexes + numIndexes, [&] { return dist(rng); });

  *query = dist(rng);

  int blockSize = 256;
  int numBlocks = (numIndexes + blockSize - 1) / blockSize;

  float time;
  cudaEvent_t start, stop;

  // printf("numIndexes: %d\n", numIndexes);
  // printf("numBlocks: %d\n", numBlocks);

  // First call does some memory stuff need to think about.
  computeDistances<<<numBlocks, blockSize>>>(numIndexes, query, indexes,
                                             distances);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  computeDistances<<<numBlocks, blockSize>>>(numIndexes, query, indexes,
                                             distances);
  
  thrust::sequence(thrust::device, keys, keys + numIndexes);

  thrust::sort_by_key(thrust::device, distances, distances + numIndexes, keys);

  cudaDeviceSynchronize();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);


  printf("Execution time:  %.3f ms \n", time);

  for (int i = 0; i < numIndexes; ++i)
    printf("%d\n", keys[i]);

  cudaFree(query);
  cudaFree(indexes);
  cudaFree(distances);

  return 0;
}