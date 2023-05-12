# GPU Accelerated Vector Database

A Proof-of-Concept vector database that supports insertion and K Nearest Neighbors search. The only supported vector is a 64-bit binary vector for simplicity of implementation and usage of space.

While this implementation can only support 64-bit binary vector, it can be extended to support any vector type, including vectors of non-binary elements such as floating point values. Such an extension would not require any alteration to the existing radix select implementation, only to the computation of distances.

In addition to supporting only 64-bit binary vectors, the only vector key, or ID, is a 16-byte UUID. Again, this was chosen for simplicity and speed of implementation. Support for arbitrarily-sized string as vector keys can be implemented easily.

## VectorDB

VectorDB is the class representing the database. VectorDB has only two methods:

1. Insert

Insert takes...

2. Query

Query takes...

Other 

VectorDB can be used either as a library (e.g., like SQLite) or it can be wrapped by a web application to serve inserts, queries, and other operations over a network. Such a web application is implemented in RestAPI.cu. 

## How it works

We used binary vectors so that...

Radix sort is a method of sorting...

Radix select is a variation that...

Our implementation of radix selection is...

Our implementation of kNN... 

## VectorDB Internals


