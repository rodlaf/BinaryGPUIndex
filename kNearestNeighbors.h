#pragma once

/*
  Used to represent 64-bit binary vector.
*/
typedef unsigned long long int uint64_cu;

/*

  All memory that is passed must be on device.
*/
void kNearestNeighbors(uint64_cu *vectors, unsigned *keys, uint64_cu *query,
                       int numVectors, int k, float *kNearestDistances,
                       uint64_cu *kNearestVectors, unsigned *kNearestKeys,
                       unsigned *workingMem1, unsigned *workingMem2,
                       unsigned *workingMem3);

#include "kNearestNeighbors.inl"