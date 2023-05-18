import uuid
import random
import time
import requests

serverURL: str = "http://ec2-54-221-48-223.compute-1.amazonaws.com"
numRequests: int = 1000
numToInsertPerRequest: int = 1000

def generateInsertBody(numToInsert: int) -> dict:
    bodyDict = {}
    bodyDict["vectors"] = []

    for i in range(numToInsert):
        id = str(uuid.uuid4())
        vector = bin(random.randint(0, 2**64 - 1))[2:]
        vectorsDict = {}
        vectorsDict["id"] = id
        vectorsDict["values"] = vector
        bodyDict["vectors"].append(vectorsDict)

    return bodyDict

# import json
# print(json.dumps(generateInsertBody(3), indent=4))

print(f"Making {numRequests} inserts with {numToInsertPerRequest} vectors per insert...")

total: int = 0
for _ in range(numRequests):
    requestBody: dict = generateInsertBody(numToInsert=numToInsertPerRequest)

    start = time.time()
    r: requests.Response = requests.post(
        serverURL + "/insert", json=requestBody
    )
    end = time.time()
    total += end - start

    assert(r.status_code == 200)

print("Total time: {:.0f} ms.".format(1000 * total))
print("Per insert average: {:.0f} ms.".format(1000 * total / numRequests))
