from openai import OpenAI
from pyexpat.errors import messages

client = OpenAI(
    api_key="sk-a692e848f05746d48bef2d57789a1de1",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    # base_url = "http://localhost:1143/v1"
)

# 2. 定义测试数据
# 格式：(文本A, 文本B, 预期标签)
examples_data = {
    "是": [
        # 语义完全一致，只是换了说法
        ("我的账号登不进去了，密码不对。", "无法登录，提示密码错误。"),
        # 核心事件一致，忽略情绪词
        ("这也太慢了，退款怎么还没到账？", "查询退款进度，一直没收到钱。"),
        # 指代同一件事
        ("iPhone 15 屏幕碎了能修吗？", "苹果手机屏幕损坏维修咨询。")
    ],
    "不是": [
        # 完全不同的主题
        ("我想买那个红色的裙子。", "这件衣服有蓝色的吗？"),
        # 相关领域，但不是同一件事
        ("怎么修改登录密码？", "忘记支付密码怎么办？"),
        # 一个是咨询，一个是投诉
        ("你们发货太慢了！", "请问今天能发货吗？")
    ]
}

# 待测问题
questions = [
    ("收不到验证码，怎么办？", "手机没收到验证码，无法注册。"),  # 预期：是
    ("如何申请发票？", "我想退货，钱什么时候退？"),      # 预期：不是
    ("App一直闪退。", "软件打开就崩溃，进不去。")        # 预期：是
]

# 3. 构建 Prompt 消息列表
messages = [
    {
        "role": "system",
        "content": "你是一个语义分析专家。请判断给定的两段文本（文本1 和 文本2）是否在描述同一件事或表达相同的意图。如果是，回答“是”；如果意图不同或主题无关，回答“不是”。"
    }
]

# 4. 动态添加 Few-Shot 示例到消息历史中
# 这一步模拟了你截图中那种“参考如下示例”的逻辑
for label, pairs in examples_data.items():
    for text_a, text_b in pairs:
        # 用户提问：给出两段文本
        messages.append({
            "role": "user",
            "content": f"文本1：【{text_a}】；文本2：【{text_b}】"
        })
        # 助手回答：给出判断结果
        messages.append({
            "role": "assistant",
            "content": label
        })

# 5. 循环发送测试问题并获取结果
print("--- 开始校验文本关联性 ---\n")

for q_a, q_b in questions:
    # 拼接当前问题
    current_user_content = f"文本1：【{q_a}】；文本2：【{q_b}】"

    # 调用 API (注意：实际生产中，为了避免上下文过长，通常不把所有历史消息都发过去，
    # 而是把 examples 拼成一个大的 system prompt，或者只保留最近几轮对话)
    response = client.chat.completions.create(
        model="qwen-turbo",  # 或者 qwen-plus / qwen-max
        messages=messages + [{"role": "user", "content": current_user_content}],
        temperature=0  # 设置为0，让结果更稳定、确定性更高
    )

    result = response.choices[0].message.content
    print(f"🔍 比对：\n   A: {q_a}\n   B: {q_b}\n   ✅ 结论: {result}\n{'-'*30}")

