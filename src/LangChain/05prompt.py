from langchain_classic.chains.summarize.map_reduce_prompt import prompt_template
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi
from dotenv import load_dotenv
import os

load_dotenv()

model = Tongyi(
    model="qwen-max",
    api_key=os.getenv("tongyi_api_key")
)

prompt_template = PromptTemplate.from_template(
    "我的邻居姓{lastname}，刚生了{gender}，你来帮我起一个名字，回答简单"
)
chain = prompt_template | model

res = chain.invoke(input={"lastname" : "张","gender" : "女孩"})

#
# prompt_text = prompt_template.format(lastname = "张",gender = "女孩")
#
# res = model.invoke(input=prompt_text)

print(res)