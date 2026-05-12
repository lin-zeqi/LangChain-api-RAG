from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms.tongyi import Tongyi
from dotenv import load_dotenv
import os

from langchain_core.runnables import Runnable, RunnableSerializable

load_dotenv()

model = Tongyi(
    model="qwen-max",
    api_key=os.getenv("TONGYI_API_KEY")
)

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system","你是一个计算机老师，专门教学java相关知识，回答十分简短"),
        ("ai","好的，我是一名老师，教学丰富生动"),
        MessagesPlaceholder("history"),
        ("human","可以告诉我java是强类型语言吗？"),
        # ......
    ]
)

history_data = [
    ("human","老师，最近学 Java 有点晕。能不能告诉我，Java 中接口（Interface）与抽象类（Abstract Class）到底有什么区别呀？感觉它们好像都能定义抽象方法，我什么时候该用哪个呢？"),
    ("ai","好的，老师来为你解答"),
]

prompt = chat_template.invoke({"history":history_data})
print(prompt)

prompt_text = prompt.to_string()
print(prompt_text)

#
chain:RunnableSerializable= chat_template | model
print(type(chain))


# print(chain.invoke({"history":history_data}))

# for chunk in chain.stream({"history":history_data}):
#     print(chunk,end='',flush=True)