#include <cstdio>
#include <chrono>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "VectorDB2.cu"

// murmur64 hash function
uint64_cu hash(uint64_cu h) {
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53L;
  h ^= h >> 33;
  return h;
}

int main(void) {

  VectorDB vdb("test.txt", 10);

  int numVectors = 10;
  const char **vectorKeys = (const char **)malloc(numVectors * sizeof(const char *));
  uint64_cu *vectors = (uint64_cu *)malloc(numVectors * sizeof(uint64_cu));
  for (int i = 0; i < numVectors; ++i) {
    vectorKeys[i] = "1234";
    vectors[i] = hash(~i);
  }

  vdb.insert(numVectors, vectorKeys, vectors);

  return 0;
}