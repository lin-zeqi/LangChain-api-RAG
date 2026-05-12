import os
from xml.dom.minidom import Document

from dotenv import load_dotenv
from huggingface_hub import search_spaces
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatTongyi(
    model="qwen3-max",
    api_key=os.getenv("TONGYI_API_KEY")
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "请严格基于我提供的参考资料（{context}）来回答，不要编造信息。如果资料里没有答案，请直接说‘资料不足’。回答要简洁、有逻辑，引用资料时请标注关键信息。"),
        ("user", "问题：{input}")
    ]
)

vector_stores = InMemoryVectorStore(
    embedding=DashScopeEmbeddings(
        model="text-embedding-v4",
        dashscope_api_key=os.getenv("TONGYI_API_KEY")
    )
)

vector_stores.add_texts(
    [
        "在Python中，变量命名应遵循蛇形命名法，即所有字母小写，单词之间用下划线分隔，例如 user_name。这有助于提高代码的可读性。",
        "函数命名同样推荐使用蛇形命名法。如果函数是私有的（仅在类内部使用），建议在名称前加一个下划线，例如 _internal_helper。",
        "类名应该使用大驼峰命名法，即每个单词的首字母大写，不使用下划线，例如 class DataProcessor:。这是 PEP 8 规范的核心建议。",
        "代码中的常量通常使用全大写字母，单词间用下划线分隔，例如 MAX_CONNECTIONS = 100。这能让维护者一眼识别出不可变的变量。",
        "在编写复杂的逻辑判断时，建议将长表达式拆分为多个有意义的布尔变量，而不是写在一行里，这样能显著提升代码的自解释性。"
    ]
)

input_text = "Python变量的命名主要是怎么个方法？"

retriever = vector_stores.as_retriever(
    search_type = "similarity", # 搜索方式
    search_kwargs = {"k" : 2}   # 最相近的两条结果
)

strparser = StrOutputParser()

# 直觉版
# chain = retriever | prompt | model | strparser()

def format_func(docs : list[Document]):
    if not docs:
        return "无相关资料"
    formatted_str = ""
    # formatted_str += "["
    for doc in docs:
        formatted_str += doc.page_content + "\n"
    # formatted_str += "]"
    return formatted_str

def print_prompt(prompt):
    print("=" * 20)
    print(prompt.to_string())
    print("=" * 20)
    return prompt

chain = (
    # 这个东西就像一个占位符，可以接收输入，可以截流
    # 链可以嵌套链
    {"input" : RunnablePassthrough(),"context" : retriever | format_func}
    | prompt
    | print_prompt
    | model
    | strparser
)

res = chain.invoke(input_text)

print(res)

# chain本质就是一个组件的输入处理后将输出传给下一个组件，使用使用chain.invoke()的时候，要注意第一个组件的输入来写