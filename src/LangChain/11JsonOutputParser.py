from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatTongyi(
    model="qwen3-max",
    api_key=os.getenv("TONGYI_API_KEY")
)

s_parser = StrOutputParser()
j_parser = JsonOutputParser()

f_prompt = PromptTemplate.from_template(
    "我的邻居姓{lastname}，刚生了{gender}，你来帮我起一个名字，回答简单,并封装为json格式返回，key就是name，value就是你起的名字，严格按照格式要求返回"
)

s_prompt = PromptTemplate.from_template(
    "姓名:{name}，请帮我解析含义"
)

chain = (f_prompt | model | j_parser | s_prompt | model | s_parser)

res = chain.invoke({"lastname":"张","gender":"女孩"})

print(res)
print(type(res))

