from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv()
model = ChatTongyi(
    model="qwen3-max",
    api_key=os.getenv("TONGYI_API_KEY")
)

f_prompt = PromptTemplate.from_template(
    "我的邻居姓{lastname}，刚生了{gender}，你来帮我起一个名字，回答简单不要说别的"
)

s_prompt = PromptTemplate.from_template(
    "姓名:{name}，请帮我解析含义"
)

s_parser = StrOutputParser()

my_func = RunnableLambda(lambda ai_msg: {"name":ai_msg.content})

chain = f_prompt | model | my_func | s_prompt | model | s_parser


for chunk in chain.stream({"lastname":"王","gender":"男孩"}):
    print(chunk,end="",flush=True)
