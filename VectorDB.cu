#include <cstdio>

#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/slice.h"

#include <thrust/sequence.h>

#include "kNearestNeighbors.h"

using ROCKSDB_NAMESPACE::DB;
using ROCKSDB_NAMESPACE::Iterator;
using ROCKSDB_NAMESPACE::Options;
using ROCKSDB_NAMESPACE::PinnableSlice;
using ROCKSDB_NAMESPACE::ReadOptions;
using ROCKSDB_NAMESPACE::Status;
using ROCKSDB_NAMESPACE::WriteBatch;
using ROCKSDB_NAMESPACE::WriteOptions;

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

__host__ void printBits(uint64_cu &x) {
  std::bitset<sizeof(uint64_cu) * CHAR_BIT> b(x);
  std::cout << b << std::endl;
}

class VectorDB {
private:
  // Pointers to device memory
  unsigned *workingMem1;
  unsigned *workingMem2;
  unsigned *workingMem3;
  uint64_cu *vectors;
  uint64_cu *queryVector;
  unsigned *deviceKeys; // sequential keys

  // Use RocksDB as persistent key-value store
  rocksdb::DB *db;
public:
  int numVectors;

  VectorDB(const std::string &name, int capacity) {
    // ROCKSDB INITIALIZATION

    Options options;

    options.IncreaseParallelism();
    options.OptimizeLevelStyleCompaction();

    // Open key value store or create it if it doesn't exist
    options.create_if_missing = true;
    Status s = DB::Open(options, name, &db);
    assert(s.ok());

    // Retrieve numVectors variable if it exists or initialize it doesn't
    std::string value;
    s = db->Get(ReadOptions(), "numVectors", &value);
    if (s.IsNotFound()) {
      s = db->Put(WriteOptions(), "numVectors", std::to_string(0));
      assert(s.ok());
      numVectors = 0;
    } else {
      numVectors = std::stoi(value);
    }

    // CUDA INITIALIZATION

    // Allocate all on-device memory
    cudaMalloc(&workingMem1, capacity * sizeof(unsigned));
    cudaMalloc(&workingMem2, capacity * sizeof(unsigned));
    cudaMalloc(&workingMem3, capacity * sizeof(unsigned));
    cudaMalloc(&vectors, capacity * sizeof(uint64_cu));
    cudaMalloc(&queryVector, sizeof(uint64_cu));
    cudaMalloc(&deviceKeys, capacity * sizeof(unsigned));

    // Initialize device keys
    thrust::sequence(thrust::device, deviceKeys, deviceKeys + numVectors);

    // Load vectors from db to device
    int iterCount = 0;
    Iterator *iter = db->NewIterator(ReadOptions());
    iter->SeekToFirst();
    for (; iter->Valid(); iter->Next(), ++iterCount) {
      assert(iterCount <= numVectors);

      // TODO: use column families so we don't have to do this kind of checking
      if (iter->key().ToString() != "numVectors") {
        uint64_cu v = std::stoull(iter->value().ToString());
        cudaMemcpy(vectors + iterCount, &v, sizeof(uint64_cu),
                   cudaMemcpyHostToDevice);
      }
    }
    delete iter;
  }

  ~VectorDB() {
    delete db;

    cudaFree(workingMem1);
    cudaFree(workingMem2);
    cudaFree(workingMem3);
    cudaFree(vectors);
    cudaFree(queryVector);
    cudaFree(deviceKeys);
  }

  /*
    Creates new key or overwrites if key already exists
  */
  void insert(const std::string &vectorKey, uint64_cu &vector) {
    // NOTE: These two should eventually be made into a transaction
    // Check if the vectorKey exists
    std::string value;
    Status getStatus = db->Get(ReadOptions(), vectorKey, &value);
    assert(getStatus.ok() || getStatus.IsNotFound());

    // Write to db and device
    Status putStatus =
        db->Put(WriteOptions(), vectorKey, std::to_string(vector));
    assert(putStatus.ok());
    cudaMemcpy(&vector, vectors + numVectors, sizeof(uint64_cu), cudaMemcpyHostToDevice);

    // Update numVectors if it is a new key
    if (getStatus.IsNotFound()) {
      Status s =
          db->Put(WriteOptions(), "numVectors", std::to_string(numVectors + 1));
      assert(s.ok());
      numVectors++;
    }
  }

  // uint64_cu& query() {

  // }
};

