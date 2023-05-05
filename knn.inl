#include <bitset>
#include <iostream>
#include <math.h>
#include <sys/time.h>

typedef unsigned long long int uint64_cu;

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