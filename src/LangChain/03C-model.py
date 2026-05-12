import os

class qwen:
    from langchain_classic.chains.question_answering.map_reduce_prompt import messages
    from langchain_community.chat_models.tongyi import ChatTongyi

    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    model = ChatTongyi(
        model="qwen3-max",
        api_key=os.getenv("TONGYI_API_KEY")
    )

    messages = [
        ("system","你是一个边塞诗人，说话简单不拖沓"),
        ("human","写一首唐诗"),
        ("ai", "锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦？"),
        ("human", "按照你之前回复的格式再写一首"),
        # SystemMessage(content="你是一个边塞诗人，说话简单不拖沓"),
        # HumanMessage(content="写一首唐诗"),
        # AIMessage(content="锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦？"),
        # HumanMessage(content="按照你之前回复的格式再写一首")
    ]
    res = model.stream(input=messages)

    for str in res:
        print(str.content, end='', flush=True)


#     本地模型
class ollama:
    from langchain_ollama.chat_models import ChatOllama
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    model = ChatOllama(
        model="qwen3-max",
        api_key=os.getenv("TONGYI_API_KEY")
    )

    messages = [
        SystemMessage(content="你是一个边塞诗人，说话简单不拖沓"),
        HumanMessage(content="写一首唐诗"),
        AIMessage(content="锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦？"),
        HumanMessage(content="按照你之前回复的格式再写一首")
    ]

    res = model.stream(input(messages))

    for str in res:
        print(str.content, end='', flush=True)