#pragma once

/*
  Used to represent 64-bit binary vector.
*/
typedef unsigned long long int uint64_cu;

/*
  TODO: return only keys and make the caller exploit the usage of keys that are
  sequential to quickly retrieve their corresponding vectors by making them
  represent array indexes; do not do that retrieval in this method.

  All memory that is passed must be on device.
*/
void kNearestNeighbors(uint64_cu *vectors, unsigned *keys, uint64_cu *query,
                       int numVectors, int k, float *kNearestDistances,
                       unsigned *kNearestKeys, unsigned *workingMem1,
                       unsigned *workingMem2, unsigned *workingMem3);

#include "kNearestNeighbors.cu"