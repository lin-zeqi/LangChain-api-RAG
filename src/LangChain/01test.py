import os
from langchain_community.llms.tongyi import Tongyi

model = Tongyi(
    model="qwen-max",
    api_key=os.getenv("TONGYI_API_KEY")
)

res = model.invoke(input="你是谁呀，你能干嘛？")

print(res);

