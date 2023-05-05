#include <bitset>
#include <iostream>
#include <math.h>
#include <sys/time.h>

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

void knn(uint64_cu *indexes, uint64_cu *query, int numIndexes, int k) {

}