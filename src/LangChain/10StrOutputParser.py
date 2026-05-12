
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatTongyi(
    model="qwen3-max",
    api_key=os.getenv("TONGYI_API_KEY")
)

prompt = PromptTemplate.from_template(
    "我的邻居姓{lastname}，刚生了{gender}，你来帮我起一个名字，回答简单"
)

parser = StrOutputParser()

chain = prompt | model | parser | model

res = chain.invoke({"lastname":"张","gender":"女孩"})

print(res.content)
print(type(res))

chain2 = chain | parser

print(chain2.invoke({"lastname":"张","gender":"女孩"}))
print(type(chain2))