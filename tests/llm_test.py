import pdb
import requests
import aiohttp
import asyncio
import json


async def llm_test():
    data = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [
        {"role": "user", "content": "create a python notebook that uses langchain to do a structured report generation for a document!"}
      ]
    }

    data_str = json.dumps(data, ensure_ascii=False)

    url = "http://localhost:8000/v1/chat/completions"
    async with aiohttp.ClientSession(json_serialize=json.dumps) as session:
        outputs = []
        for i in range(10):
            outputs.append(session.post(url, json=data))
        out = await asyncio.gather(*outputs, return_exceptions=True)
    return out


outputs = asyncio.run(llm_test())
breakpoint()



data = {
    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "messages": [
    {"role": "user", "content": "create a python notebook that uses langchain to do a structured report generation for a document!"}
  ]
}
requests.post("http://localhost:8000/v1/chat/completions", data=data)
print(json.loads(ass.text)["choices"][0]["message"]["content"])



client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)
completion = client.chat.completions.create(
  model="mistralai/Mixtral-8x7B-Instruct-v0.1",
  messages=[
    {"role": "user", "content": "create a python notebook that uses langchain to do a structured report generation for a document!"}
  ]
)
print(completion.choices[0].message)

# docker run --runtime nvidia --gpus all \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
#     -p 8000:8000 \
#     --ipc=host \
#  vllm/vllm-openai:latest \
#  --model mistralai/Mistral-7B-Instruct-v0.2 --dtype half --gpu-memory-utilization 1


# $MIG_DEVICES="device=MIG-f665593c-0d36-5f04-80df-b357a0e298ca,MIG-fd165a8a-0209-57a4-80d0-0484e95589d9,MIG-d5a923ad-e0a3-53df-adcb-3ecb8bc801cb,MIG-a9aa7947-3a10-5646-8da6-3c2d820b95d6"
# docker run --runtime nvidia --gpus  '$MIG_DEVICES'   \
#     -v ~/.cache/huggingface:/root/.cache/huggingface     --env "HUGGING_FACE_HUB_TOKEN=<secret>"     -p 8000:8000     --ipc=host  vllm/vllm-openai:latest  --model mistralai/Mixtral-8x7B-Instruct-v0.1 --dtype half --gpu-memory-utilization 1 --worker-use-ray --tensor-parallel-size 4

# nvidia-smi -i 0 -mig 1
# nvidia-smi -i 1 -mig 1
# nvidia-smi -i 2 -mig 1
# nvidia-smi -i 3 -mig 1
# nvidia-smi --gpu-reset

# nvidia-smi mig -cgi "5,19,19,19" -C
# nvidia-smi -L

# # nvidia-smi mig -dci && sudo nvidia-smi mig -dgi
