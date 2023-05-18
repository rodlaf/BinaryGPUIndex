#pragma once

/*
  Used to represent 64-bit binary vector.
*/
typedef unsigned long long int uint64_cu;

/*
  keys MUST be a sequence of integers representing array indexes!!
  e.g., [0, 1, 2, ..., numVectors].

  All memory that is passed must be on device.
*/
void kNearestNeighbors(uint64_cu *vectors, unsigned *keys, uint64_cu *query,
                       int numVectors, int k, float *kNearestDistances,
                       unsigned *kNearestKeys, unsigned *workingMem1,
                       unsigned *workingMem2, unsigned *workingMem3);

#include "kNearestNeighbors.cu"