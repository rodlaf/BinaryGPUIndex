#include <iostream>
#include <fstream>

#include "kNearestNeighbors.h"

// Design decision: Seperate deviceKeys and vectorKeys. 
// 
// vectorKeys can be very long (e.g., a whole UUID) but deviceKeys must
// be a single unsigned integer (e.g., 32 bits). This imposes a limitation
// that at most 2^32 billion vectors can be searched on a GPU at a time, no 
// matter the memory constraints of the GPU. This can be changed in the future.
//
// a separate collection of deviceKey-to-vectorkey key-value pairs will be kept
// in either CPU memory or the persistent key value store (e.g., RocksDB).
//
// in addition, deviceKeys will be sequential in order to enable quick vector 
// retrieval by interpreting them as indexes in the on-device vector array 

class VectorDB {
private:
  // Pointers to device memory
  unsigned *workingMem1;
  unsigned *workingMem2;
  unsigned *workingMem3;
  uint64_cu *vectors;
  uint64_cu *queryVector;
  unsigned *seqKeys; // sequential keys

public:
  int count = 0;

  VectorDB() {
    vectorsFile.open("test.txt");
  }

  ~VectorDB() {
    vectorsFile.close();
  }

  void insert(int key, uint64_cu vector) {
    printf("count: %d\n", count);

  }

  void query() {

  }
};

int main(void) {
  VectorDB vdb;

  // vdb.insert();

  return 0;
}