# Cuda Accelerated Vector kNN

A simple vector database that supports insertion and K Nearest Neighbors search. The only supported vector is a 64-bit binary vector, for simplicity of implementation and usage of space.

While this implementation can only support 64-bit binary vector, it can be extended to support any vector type, including vectors of non-binary elements such as floating point values. Such an extension would not require any alteration to the existing radix select implementation, only to the computation of distances.

## VectorDB

VectorDB is...

## How it works

We used binary vectors so that...

Radix sort is a method of sorting...

Radix select is a variation that...

Our implementation of radix selection is...

Our implementation of kNN... 


