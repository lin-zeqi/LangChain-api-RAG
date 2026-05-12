import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("TONGYI_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

responce = client.chat.completions.create(
    model="qwen3-max",

    messages=[
        {"role": "system" , "content": "你是一个Python专家，话特别多"},
        {"role": "assistant" , "content": "好的，我是python专家，话特别多"},
        {"role": "user" , "content": "计算1+2一直加到10，使用python代码实现 "}
    ],

    stream=True
)

for chunk in responce:
    print(
        chunk.choices[0].delta.content,
        end=' ',
        flush=True
    )