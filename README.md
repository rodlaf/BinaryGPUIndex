# GPU Vector Index

A proof of concept vector index that supports insertion and k-nearest-neighbors querying. The index is implemented as DeviceIndex in `DeviceIndex.cu` and can be used as a library within a C++ program. A simple web server that allows insertions and queries over a network is implemented in `server.cu`.

While this implementation can only support 64-bit binary vectors, it can be extended to support any vector type including vectors of non-binary elements such as floating point values. Such an extension would only change the way in which distances are computed between vectors and not the way in which those distances are ranked (e.g., radix select).

Vectors must be inserted along with an UUID. This index can be extended to support other types of vector keys, including arbitrarily-sized strings. UUIDs were chosen for their frequent use and because on a technical level, it was easier to store them given they are of constant length.

## Dependencies

- Cuda Toolkit 11.0+
- Crow 1.0+

## Usage

The server is complied and run in the following manner, in a shell:

    nvcc server.cu
    ./a.out [index filename]

[index filename] must be replaced by the name of the file where indexes are 
to be stored. 

The server will then specify the port on which it is running, which should be configured in the appropriate manner if using a cloud service to host (e.g., AWS EC2).

The server has two methods, explained below:

- POST /query
- POST /insert

## Inserting

Requests to /insert must contain, in the body, the vectors to be inserted, along with their ids, in JSON format. The format is the following:

```json
{
    "vectors": [
        {
            "id": "[valid UUID]"
            "values": "[valid binary string of length 64]"
        },

        ...
    ]
}
```

The following is an example of a valid request to /insert:

```json
{
    "vectors": [
        {
            "id": "4d1027ec-80b7-4df3-b950-ae824fadbd61",
            "values": "1000001011111100100011010001100010011000110010011110110111110110"
        },
        {
            "id": "e78241cc-5bc6-4532-8b7c-76809c2704bd",
            "values": "110010110000101101011000101101110101000110110010001100001110010"
        },
        {
            "id": "87e298cc-8e46-4a6a-922c-127026f99dea",
            "values": "100010101010101100000011101101000000011011010100000000001110001"
        }
    ]
}
```

The response will return the number of vectors inserted, if succesfull. The response to the example request above would be 

```json
{
    "insertedCount": "3"
}
```

## Quering

Request to /query must contain the vector to be queried and the number topK of vectors to be retrieved. The format is the following:

```json
{
    "topK": "[valid integer]",
    "vector": "[valid binary string of length 64]"
}
```

The topK amount must be less than or equal to the number of vectors in the index for the query to succeed.

The following is an example of a valid /query POST body:

```json
{
    "topK": "1000",
    "vector": "1100111111101100111100110010111011000101000001011101010010010100"
}
```

The response is a list of the retrieved vectors (matches) along with their ids and corresponding distances, which represent cosine distances. The format is the following:

```json
{
    "matches": [
        {
            "values": "[binary string representing vector]",
            "distance": "[floating point value]",
            "id": "[vector UUID]"
        },

        ...
    ]
}
```

The following is an example response to a query with topK equal to 3:

```json
{
    "matches": [
        {
            "values": "1100111111101100111100110010111011000101000001011101010010010100",
            "distance": "0.125980",
            "id": "8ea44221-707e-4b26-815a-90bb60339401"
        },
        {
            "id": "770d9f87-7a81-484d-95f9-5ca3321a6028",
            "distance": "0.125980",
            "values": "1100110101101010111100110010111011000111000101011111110011010100"
        },
        {
            "values": "1100111111001100111000010010111111000001000001111110010000010100",
            "distance": "0.137542",
            "id": "2c7a0a4f-de42-482a-9bee-a9f4c3c3102b"
        }
    ]
}
```

## Benchmark 

The results of simple benchmark on an index containing half a billion vectors is 
shown below.

    Opening index...
    Done. Execution time: 195572 ms.
    Server is running on port 80.

Opening the index containing 500 million indexes took a little over 3 minutes. This is a one-time wait for the entire duration of the server, and this opening must be done every time the server is killed and started again. 

A benchmark of 1000 queries shows an average latency of 193 milliseconds:

    Making 1000 queries with topK=1000...
    Total time: 193152 ms.
    Per query average: 193 ms.

As for inserts, the following is a benchmark on inserts of 1000 vectors at a time.

    Making 1000 inserts with 1000 vectors per insert...
    Total time: 114483 ms.
    Per insert average: 114 ms.


## Future directions

- Implementation of insert as upsert, e.g., better behavior for insertion of keys that already exist. In the current implementation, duplicates of ids are allowed.
