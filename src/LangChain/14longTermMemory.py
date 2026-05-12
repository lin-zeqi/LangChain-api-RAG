import json
import os
from typing import Sequence
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
# 以下是新增的导入,前者完成的是将单个消息对象转变为字典；后者完成的是将包含多个字典的列表转变为包含多个消息的列表
from langchain_core.messages import message_to_dict, messages_from_dict, BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory



# 继承BaseChatMessageHistory
class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self,session_id,storage_path):
        self.session_id = session_id    # 会话的id
        self.storage_path = storage_path    # 不同会话id的存储文件所在的文件夹路径

        # 拼接为完整的文件路径，调用os中的路径拼接方法
        self.file_path = os.path.join(self.storage_path,self.session_id)

        # 首先确保文件所在的文件夹路径存在，如果不存在就创建，存在就跳过，确保不会报错
        os.makedirs(os.path.dirname(self.file_path),exist_ok=True)

    # 可以将Sequence序列当作是list这种，而BaseMessage则是AIMessage等message的父类
    def add_message(self,message: BaseMessage) -> None:
        all_messages = list(self.messages)
        all_messages.append(message)

        # 就数据同步写入本地文件中去
        # 类对象写入文件得到二进制文件，所以需要将BaseMessage转变为字典，然后通过json模块将其转为json写入文件
        # 然后使用message_to_dict 来实现单个消息对象转字典
        # 下面有两周写法，采用第二种写法会更加清晰
        # new_messages = []
        # for message in all_messages:
        #     d = message_to_dict(message)
        #     new_messages.append(d)

        new_messages = [message_to_dict(message) for message in all_messages]
        #将数据写入文件
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump(new_messages,f)

    @property # @propery装饰器，将messages方法变为成员变量属性
    def messages(self) -> list[BaseMessage]:
        try:
            with open(self.file_path,"r",encoding="utf-8") as f:
                messages = json.load(f)
                return messages_from_dict(messages)
        except FileNotFoundError:
            return []

    def clear(self) -> None:
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump([],f)


load_dotenv()
model = ChatTongyi(
    model="qwen3-max",
    api_key=os.getenv("TONGYI_API_KEY")
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","你需要根据历史会话记录来回应用户问题。对话历史："),
        (MessagesPlaceholder("chat_history")),
        ("human","请回答如下问题：{input}")
    ]
)
str_parser = StrOutputParser()

store = {}

def get_history(session_id):
    return FileChatMessageHistory(session_id,"./chat_history")

def print_prompt(full_prompt):
    print("="*20,full_prompt.to_string(),"="*20)
    return full_prompt


base_chain = prompt | print_prompt | model | str_parser

chain = RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

if __name__ == "__main__":
    session_user = {
        "configurable" : {
            "session_id" : "user001"
        }
    }

    res1 = chain.invoke({"input":"小明有两只猫"}, session_user)
    print("1",res1)

    res2 = chain.invoke({"input":"小红有三只狗"}, session_user)
    print("2", res2)

    # res3 = chain.invoke({"input":"一共有多少只动物"}, session_user)
    # print("3", res3)


