import os
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
# 以下是新增的导入
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()
model = ChatTongyi(
    model="qwen3-max",
    api_key=os.getenv("TONGYI_API_KEY")
)

# 这里有两种提示词模板的方式，第一种是通用提示词，第二种是聊天提示词，整体用下来第二种更好一些

# prompt = PromptTemplate.from_template(
#     "你需要根据历史会话记录来回应用户问题。对话历史：{chat_history}，用户提问：{input}，请回答"
# )
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","你需要根据历史会话记录来回应用户问题。对话历史："),
        # 这个在之前聊天提示词模板的地方用过
        (MessagesPlaceholder("chat_history")),
        ("human","请回答如下问题：{input}")
    ]
)
# 初始化解析器
str_parser = StrOutputParser()
# 存储不同用户的不同聊天历史
store = {}
# 自定义函数，用于根据用户id获取其聊天历史记录
def get_history(session_id):
	# 判断是否有该id的用户，没有就将其记录加入
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
# 自定义函数，不影响整体运行，为了更加方便观看提示词情况
def print_prompt(full_prompt):
    print("="*20,full_prompt.to_string(),"="*20)
    return full_prompt

# 构建基础链
base_chain = prompt | print_prompt | model | str_parser

# 使用组件，将基础链包装成带记忆的链
chain = RunnableWithMessageHistory(
    base_chain,		# 基础链
    get_history,	# 获取历史记忆
    input_messages_key="input",		# 用户输入的变量名
    history_messages_key="chat_history"		#历史记忆的变量名
)
# 测试代码：演示多轮对话的记忆功能
if __name__ == "__main__":
	# 这里是 RunnableWithMessageHistory 要求的特定格式
    session_user = {
        "configurable" : {		# 特定键名，不可改变
            "session_id" : "user001"		# 用户id
        }
    }
	# 注意：invoke方法在RunnableWithMessageHistory中被增强，增加了自动记忆管理功能
    res1 = chain.invoke({"input":"小明有两只猫"}, session_user)
    print("1",res1)

    res2 = chain.invoke({"input":"小红有三只狗"}, session_user)
    print("2", res2)

    res3 = chain.invoke({"input":"一共有多少只动物"}, session_user)
    print("3", res3)

