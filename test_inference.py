import pdb
import jsonpickle
import requests
from torchvision.io import read_image
import aiohttp
import asyncio

def non_batch_test():
    img = read_image("/data/chris/model-servers/lena.png")
    data = {"data": jsonpickle.encode(img)}
    # data = [{"data": jsonpickle.encode(img)}, {"data": jsonpickle.encode(img)}, {"data": jsonpickle.encode(img)}]
    outputs = []
    for i in range(4):
        outputs.append(requests.post("http://localhost:8780/predictions/test-model-low", data=data))

async def main():
    async with aiohttp.ClientSession() as session:
        for number in range(1, 151):
            pokemon_url = f'https://pokeapi.co/api/v2/pokemon/{number}'
            async with session.get(pokemon_url) as resp:
                pokemon = await resp.json()
                print(pokemon['name'])


async def batch_test():
    img = read_image("/data/chris/model-servers/lena.png")
    data = {"data": jsonpickle.encode(img)}

    url = "http://localhost:8780/predictions/test-model"
    async with aiohttp.ClientSession() as session:
        outputs = []
        for i in range(5):
            outputs.append(session.post(url, data=data))
        out = await asyncio.gather(*outputs, return_exceptions=True)
    return out
# outputs = non_batch_test()
outputs = asyncio.run(batch_test())
pdb.set_trace()
# output = jsonpickle.decode(output.json()["output"])

# models={\
#   "test-model": {\
#     "1.0": {\
#         "defaultVersion": true,\
#         "marName": "test-model.mar",\
#         "minWorkers": 1,\
#         "maxWorkers": 1,\
#         "batchSize": 4,\
#         "maxBatchDelay": 4000,\
#         "responseTimeout": 5200\
#     }\
#   }\
# }