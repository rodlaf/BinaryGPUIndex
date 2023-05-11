#include <cstdio>
#include <chrono>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/slice.h"

#include "VectorDB.cu"

// __host__ void printBits(uint64_cu &x) {
//   std::bitset<sizeof(uint64_cu) * CHAR_BIT> b(x);
//   std::cout << b << std::endl;
// }

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
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;

  const std::string vdbName = "/tmp/testdb";
  int a10gCapacity = 1 << 20;
  int numVectorsToInsert = 10000;

  auto t1 = high_resolution_clock::now();
  VectorDB vdb(vdbName, a10gCapacity);
  auto t2 = high_resolution_clock::now();

  duration<double, std::milli> ms_double = t2 - t1;
  std::cout << ms_double.count() << "ms\n";

  // insert vectors
  t1 = high_resolution_clock::now();
  for (int i = 0; i < numVectorsToInsert; ++i) {
    // generate uuid as vector key
    boost::uuids::uuid uuid = boost::uuids::random_generator()();
    const std::string vKey = boost::uuids::to_string(uuid);

    // generate random vector using hash function
    uint64_cu v = hash(~i);
    
    // insert vector
    vdb.insert(vKey, v);
  }
  t2 = high_resolution_clock::now();
  ms_double = t2 - t1;
  std::cout << ms_double.count() << "ms\n";

  printf("numVectors: %d\n", vdb.numVectors);

  // query
  const int k = 10;
  uint64_cu queryVector = hash(~1);
  uint64_cu *kNearestVectors = (uint64_cu *)malloc(k * sizeof(uint64_cu));
  float *kNearestDistances = (float *)malloc(k * sizeof(float));
  std::string kNearestVectorKeys[k];

  vdb.query(&queryVector, k, kNearestDistances, kNearestVectors, kNearestVectorKeys);

  // print results
  printf("Query: ");
  printBits(queryVector);
  for (int i = 0; i < k; ++i) {
    printf("%d: %s %8.8f ", i, kNearestVectorKeys[i].c_str(), kNearestDistances[i]);
    printBits(kNearestVectors[i]);
  }

  free(kNearestVectors);

  return 0;
}