import os
from openai import OpenAI
from pyexpat.errors import messages

client = OpenAI(
    api_key=os.getenv("TONGYI_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

system_content = "你是一个前台接待助手。请从用户输入中提取 JSON 格式的信息，包含字段：name(姓名), phone(电话), purpose(来访目的)。"

examples = [
    {"role": "user", "content": "你好，我是张三，电话13800138000，我想预约明天的面试。"},
    {"role": "assistant", "content": '{"name": "张三", "phone": "13800138000", "purpose": "面试"}'},

    {"role": "user", "content": "李四，13912345678，过来修电脑的。"},
    {"role": "assistant", "content": '{"name": "李四", "phone": "13912345678", "purpose": "维修电脑"}'},
]

new_question = "我是王五，号码是13666668888，来找你们老板谈合作。"

messages= [
    {"role": "system", "content": system_content}
]

messages.extend(examples)



print(messages)


responce = client.chat.completions.create(
    model="qwen3-max",
    messages=messages + [
        {"role": "user", "content": new_question}
    ]
)

print(responce.choices[0].message.content)