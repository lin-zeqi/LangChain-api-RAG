import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("TONGYI_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

responce = client.chat.completions.create(
    model="qwen3-max",

    messages=[
        {"role": "system" , "content": "你是一个ai助理，话不多"},
        {"role": "user" , "content": "小明有三个鸡蛋"},
        # {"role": "assistant" , "content": "好的"},
        {"role": "user" , "content": "小红有五个鸡蛋"},
        # {"role": "assistant" , "content": "好的"},
        {"role": "user", "content": "小明和小红一共有多少个鸡蛋"},
    ],

    stream=True
)

for chunk in responce:
    print(
        chunk.choices[0].delta.content,
        end=' ',
        flush=True
    )