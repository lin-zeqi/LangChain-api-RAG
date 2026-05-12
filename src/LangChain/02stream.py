import os
from langchain_community.llms.tongyi import Tongyi

model = Tongyi(
    model="qwen-max",
    api_key=os.getenv("TONGYI_API_KEY")
)

res = model.stream(input="你是谁呀")

for str in res:
    print(str,end='',flush=True)

# 本地调用流式输出

from langchain_ollama import OllamaLLM

models = OllamaLLM(
    model="ollama3.1",
)

res = model.stream(input="你是谁呀")

for str in res:
    print(str,end='',flush=True)
