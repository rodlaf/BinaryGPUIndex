#include <cstdio>

#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/slice.h"

#include "VectorDB.cu"

int main(void) {
  int a10gCapacity = 970000000; // Experimentally computed

  VectorDB vdb("/tmp/testdb", a10gCapacity);
  printf("numVectors: %d\n", vdb.numVectors);

  const std::string vKey = "someRandomkeyThatCanBeAnythingAndLong23";
  // uint64_cu v =
  //     0b0000111100001111000011110000111100001111000011110000111100001111;
  uint64_cu v =
      0b0000111100001111000011110000111100001111000011110000111100001111;

  vdb.insert(vKey, v);

  return 0;
}