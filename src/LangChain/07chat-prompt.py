from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms.tongyi import Tongyi
from dotenv import load_dotenv
import os


load_dotenv()

model = Tongyi(
    model="qwen-max",
    api_key=os.getenv("TONGYI_API_KEY")
)

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system","你是一个计算机老师，专门教学java相关知识"),
        ("ai","好的，我是一名老师，教学丰富生动"),
        MessagesPlaceholder("history"),
        ("human","可以告诉我java是强类型语言吗？"),
        # ......
    ]
)

history_data = [
    ("human","老师，最近学 Java 有点晕。能不能告诉我，Java 中接口（Interface）与抽象类（Abstract Class）到底有什么区别呀？感觉它们好像都能定义抽象方法，我什么时候该用哪个呢？"),
    ("ai"," 哈哈，这个问题问得好！这可是 Java 面试里的“必考题”，很多初学者都会被绕晕。来，老师给你打个最通俗的比方，保证你一听就懂！"),
]

prompt = chat_template.invoke({"history":history_data})
print(prompt)

prompt_text = prompt.to_string()
print(prompt_text)

chain = chat_template | model
print(chain.invoke({"history":history_data}))