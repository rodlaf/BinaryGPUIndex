# GPU Accelerated Vector Index

A proof of concept vector index that supports insertion and k-nearest-neighbors querying. The index is implemented as DeviceIndex in `DeviceIndex.cu` and a simple web server that allows insertions and queries over a network is implemented in `server.cu`.

While this implementation can only support 64-bit binary vectors, it can be extended to support any vector type including vectors of non-binary elements such as floating point values. Such an extension would only change the way in which distances are computed between vectors and not the way in which those distances are ranked (e.g., radix select).

Vectors must be inserted along with an UUID. This index can be extended to support other types of vector keys, including arbitrarily-sized strings. UUIDs were chosen for their frequent use and because on a technical level, it was easier to store them given they are of constant length.

## Dependencies

- Cuda Toolkit 11.0+
- Crow 1.0+

## Usage

The implementation can be tested without use of a network by running

    nvcc testDeviceIndex.cu
    ./a.out

This test will generate random vectors using a hash function and insert them 
into a new DeviceIndex. A query will be made, and then the DeviceIndex will 
be deleted. Following this, a new, different DeviceIndex will be made using the
file created by the old one. The same query as before will be made, and the 
results will be shown to test that they are equal. 

## Benchmark 

The results of simple benchmark on a single query of half a billion vectors is 
shown below.

    Opening...
    numVectors: 500000000
    Done. Execution time: 195310ms.
    Querying...
    Done. Execution time: 52ms.
    k = 10
    Query: 0011101010000101100100111000100001101100010101011010000000101011
    0: 5b9dda93-977b-414e-91de-68ad617b4ab3 0.00000000 0011101010000101100100111000100001101100010101011010000000101011
    1: 83da579b-7721-44b9-8bb9-0e64ef46a5c6 0.17487699 0011101011100101101101111001110001000100010101111010000110101011
    2: 9091f50d-b508-40c8-82e5-061a1ec95357 0.18350339 1011101010011101100101111000110011001100011100011010000000100111
    3: c422d8c1-78fe-4b62-9e8b-b4cb0ca5fea1 0.17487699 1011001010000101100111111010100001101100111101001110000100101111
    4: 57321faf-6bcb-4d53-be4d-edb8b03774a6 0.17487699 0011101011100101101101111001110001000100010101111010000110101011
    5: 964cfd4f-37b3-4feb-85fb-f13ebf29e794 0.18350339 0011101011000000100100111110100001111101010111011110000001100011
    6: a80fd471-d49b-4fad-80aa-3baf780bffd1 0.17487699 1011001010000101100111111010100001101100111101001110000100101111
    7: bfb98c28-2826-4199-b2e6-e46c7cf2f154 0.17487699 1011001010000101100111111010100001101100111101001110000100101111
    8: 97f3a56c-c291-4ad3-a6b1-508b3b03a193 0.18829232 0111101010100111110100111010110001101000010101111010101011101111
    9: c4ddbfa8-8937-4bf5-a345-5f919119713d 0.18829232 0011101011010101110110111010000011111100110111011110010000111011


## DeviceIndex 🚧

DeviceIndex is the class representing the database. DeviceIndex has only two methods:

1. Insert

Insert takes...

2. Query

Query takes...

## How it works 🚧

We used binary vectors so that...

Radix sort is a method of sorting...

Radix select is a variation that...

Our implementation of radix selection is...

Our implementation of kNN... 

