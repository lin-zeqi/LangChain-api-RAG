# LangChain

​	LangChain 是一个用于构建大语言模型应用的开源框架，核心思想是把「模型」「提示词」「数据」「记忆」「工具」等组件像搭积木一样串联成可复用的工作流（Chain）。
​	其设计哲学是「组件化 + 链式调用」，每个模块职责单一、接口标准（Runnable），既能快速搭建原型，也便于生产级扩展。

> 该文档出自笔者自身学习 LangChain 的实战笔记，内容基于官方文档、社区教程及个人项目踩坑经验整理而成。文中代码示例均以阿里云百炼（DashScope）为后端，技术选型偏向 Java/Python 混合架构场景。因大模型生态迭代迅速，部分 API 或参数可能随版本更新而变化，建议结合实际环境验证使用。如有疏漏，欢迎交流指正～ 

​	全文以 **「组件解耦 + 链式组装」** 为脉络，由浅入深划分为以下核心模块：

- ​	**核心模型调用**：拆解大语言模型（LLMs）、对话模型（Chat Models）与嵌入模型（Embeddings）的底层差异，详解同步/流式调用规范及消息对象处理
- ​	**提示词模板工程**：对比通用模板、少样本提示（Few-Shot）与多轮对话模板的适用场景，厘清 `format()` 字符串替换与 `invoke()` 标准接口的调用边界
- ​	**链式工作流与 Runnable 接口**：揭示 `|` 运算符串联组件的底层逻辑与数据流转契约
- ​	**输出解析与灵活转换**：通过 Str/Json 解析器与 RunnableLambda 解决模型非结构化输出的格式化难题
- ​	**对话记忆管理**：从内存短期缓存延伸至文件级长期持久化，保障多轮交互的连贯性
- ​	**文档加载与文本切分**：整合 CSV、JSON、PDF 及纯文本加载器，配合递归分割器完成非结构化数据向 Document 对象的标准转化
- ​	**向量存储与 RAG 检索链路**：对比内存与 Chroma 外部数据库的持久化方案，利用 `as_retriever()` 适配检索器，并引入 RunnablePassthrough 构建“问题截流+上下文注入”的并行数据流，完整落地检索增强生成（RAG）的工业级闭环

​	希望我的学习笔记可以帮助你对 LangChain 快速的建立一个属于你自己的知识体系



## Models
​	LangChain主要支持的模型主要有三种类型：LLMs（大语言模型）、Chat Models（聊天模型）、Embeddings Models（嵌入模型）
​		LLMs 是技术范畴的统称，一般是指基于大量参数训练出来的TransFormer架构的模型，主要能力就是理解和生成自然语言
​		Chat Models 主要是指专用于聊天场景的 LLMs，主要能力就是模拟和人类的对话
​		Embedding Model 就是我们RAG中的文本嵌入模型，接收文本作为输入，然后进行向量化后得到文本向量
​	注意哦，有些模型名字很像，但不一定是一样的哦，比如千问的 qwen3-max 和 qwen-max ，前者是聊天模型，后者是大语言模型，然后就是聊天模型和大模型之间的调用的，在一般地方是差不多的，关键在于输出的时候，聊天模型循环输出的时候需要使用 .content() 方法，使用该方法目的在于提取元数据中的文字部分，如果不用的话，看到的可能就是一堆原始数据

### 模型调用
​	**大语言模型**的调用首先需要引入对应模块，例如：

```py
from langchain_community.llms.tongyi import Tongyi	
```

​	随后将 Tongyi 对象中填入参数（model和apikey）然后赋值给一个变量，让其成为对象：
```py
model = Tongyi(
    model="qwen-max",
    api_key="sk-********************************" # 这里要填入你的api—key
)
```

​	该对象可以以两种形式进行输出：
```py
# 非流式输出
res = model.invoke(input="你是谁呀，你能干嘛？")
```

​	和
```py
# 流式输出
res = model.stream(input="你是谁呀")
```

​	而需要注意的一点是，面对流式输出的时候，其输出方式并非直接的使用 print 打印出来，因流式输出返回的是一个迭代器，所以需要进行循环输出：
```py
for str in res:
# end表示以什么结尾，flush表示是否刷新
    print(str,end='',flush=True)
```



​	**聊天模型**的调用在引用模块方面就不大一样：

```py
# 第一个导包引用的是聊天模型
from langchain_community.chat_models.tongyi import ChatTongyi
# 第二个导包引用的是langchain中的message
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
```

​	随后的对象操作和大语言模型一致，而不同的点在于这个 message 这一块，主要用于创建类对象
​	聊天模型的提问和回答相较于大语言模型而言，更带有角色标签的性质，可以让ai明确的知道每一句话是谁说的
​		AIMessage 就是 AI 输出的消息，可以是针对问题的回答
​		HumanMessage 就是人类消息也可以称之为用户信息，由人给出的信息发送给 LLMs 的提示信息或问题
​		SystemMessage 可以用于指定模型具体所处的环境和背景，可以让其进行角色扮演等。比如 “作为一个代码专家” 或 “返回json格式” 

```py
# 用列表包含每一个Message，这是静态的一步到位，直接得到Message类对象
messages = [
        SystemMessage(content="你是一个边塞诗人，说话简单不拖沓"),
        HumanMessage(content="写一首唐诗"),
        AIMessage(content="锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦？"),
        HumanMessage(content="按照你之前回复的格式再写一首")
    ]
```

​	其余部分的调用与大模型调用一致，仅在 res 处需要将 message 这个列表传给 model ：
```py
res = model.stream(input=messages)
```

​	但针对聊天模型的 message ，还有一种简写方式：
```py
# 列表内不再是类对象，而是元组，且不再需要导入类对象的包（只有这三个角色），这是动态的，需要在运行时LangChain转化为类对象
# 还支持在内部填充变量，使用{变量}占位符
messages = [
        ("system","你是一个边塞诗人，说话简单不拖沓"),
        ("human","写一首唐诗"),
        ("ai", "锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦？"),
        ("human", "按照你之前回复的格式再写一首")
    ]
```



​	**文本嵌入模型**（Embedding Models）的调用也需要引入不同的模块：

```py
# 这里引入的是阿里云的嵌入模型
from langchain_community.embeddings import DashScopeEmbeddings
```

​	随后对其进行初始化对象：
```py
# 这里如果不写模型是什么的话，也是默认使用这个模型的
embed = DashScopeEmbeddings(
    model='text-embedding-v1',
    dashscope_api_key="sk-********************************"  # 还是你的api—key，上下两个都是的
)
```

​	关键点在于这个对象的输出
​	embedding对象的输出不同于调用模型的流式输出和非流式输出，其在与将文本转化为向量，也就是文本的向量化
​	所以，该模型接收的对象是文本，输出的是一个向量列表：

```py
# 这里的 embed_query 可以理解为将单个字符串进行转换
print(embed.embed_query("我喜欢你"))
print(embed.embed_query("我爱你"))
# 这里的 embed_documents 可以理解为将多个字符串进行转换
print(embed.embed_documents(['我喜欢你','我稀饭拟'],['晚上吃什么']))
```



## Promopt 模板

###  通用Prompt模板
​	Prompt Template 类表示提供一个模板用于构建基础模板，支持变量注入，生成完整所需提示词：

```py
# 例如以这种形式进行注入，使用 .format() 方法来实现
prompt_template = PromptTemplate.from_template(
    "我的邻居姓{lastname}，刚生了{gender}，你来帮我起一个名字，回答简单"
)

prompt_text = prompt_template.format(lastname = "张",gender = "女孩")
```

​	但是也可以直接注入，比如：
```py
lastname = "张"
gender = "女孩"
s = f"我的邻居姓{lastname}，刚生了{gender}，你来帮我起一个名字，回答简单"
```

​	依靠这种方法虽然在实际中可以操作，但不符合 Runnable 接口的规范，调用该接口仅可以通过上方的 Prompt Template 实现，该接口可以将 prompt 转变并支持加入到 LangChain 中的 Chains 这个组件中去，单纯的字符串是无法加入的
​	而 Chain 的简单链式调用如下：

```py
prompt_template = PromptTemplate.from_template(
    "我的邻居姓{lastname}，刚生了{gender}，你来帮我起一个名字，回答简单"
)
# 这里就是将prompt_template直接传入model中去，| 表示传入，以这种方式形成一个链条，字符串不可以传入
chain = prompt_template | model
# 而这里的注入必须要是以字典的形式实现
res = chain.invoke(input={"lastname" : "张","gender" : "女孩"})
```

### 少样本提示词模板
​	FewShotPromptTemplate，在概念上与通用提示词样本的不同可能仅在于将提示词与少量样本结合为完整提示词并传入，但在写法上还是有较大的不同的

```py
# 示例的模板
example_template = PromptTemplate.from_template("单词:{word}，反义词:{antonym}")

# 示例的数据
example_data =[
    {"word":"大","antonym":"小"},
    {"word":"上","antonym":"下"}
]

# 使用少样本
few_shot = FewShotPromptTemplate(
    example_prompt = example_template,
    examples = example_data,		# 这里要注意，必须是列表嵌套字典
    prefix = "告知我单词的反义词，我提供以下示例",		#这里是在示例之前写入的提示词
    suffi = "基于示例，告知我这个单词{input_word}反义词是什么?",		#这里是在示例之后写入的提示词
    input_variables=['input_word']		#声明需要注入的变量是什么
)
```

### 聊天提示词模板
​	ChatPromptTemplate 是一种支持注入任意数量历史信息的提示词模板，该提示词模板可以通过使用from_messages方法获取多轮会话作为背景

```py
ChatPromptTemplate.from_messages(
    [
        ("system","你是一个计算机老师，专门教学java相关知识"),
        ("ai","好的，我是一名老师，教学丰富生动"),
        ("human","可以告诉我java中接口与抽象类的区别吗？"),
        # ......
    ]
)
```

​	上面这种引用是不是和之前学习 ChatModel 部分的 message 很类似？也和通用模板中的 from_template 也很相似？那使用这个模板的区别在哪？
​	区别就在于其对通用模板而言其不只是如上所示的接入一条消息，其在列表中可以引入多种会话信息。而对于 message 而言它的区别在于可以动态注入，就是可以根据时间历史来动态的进行
​	这上面的这种仅是最基本的使用方式，下面这一种算是更牛逼一点的：

```py
# 聊天模板的定义，其中引入一个MessagesPlaceholder方法，用来将注入的历史记录放到会话中
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system","你是一个计算机老师，专门教学java相关知识"),
        ("ai","好的，我是一名老师，教学丰富生动"),
        # 这是一个类，在此处创建类对象，而这里的history是key，对应后面注入的value
        MessagesPlaceholder("history"),
        ("human","可以告诉我java中接口与抽象类的区别吗？"),
        # ......
    ]
)

# 历史会话记录的部分，可以动态的修改这个列表
history_data = [
    ("human","老师，最近学 Java 有点晕。能不能告诉我，Java 中接口（Interface）与抽象类（Abstract Class）到底有什么区别呀？感觉它们好像都能定义抽象方法，我什么时候该用哪个呢？"),
    ("ai"," 哈哈，这个问题问得好！这可是 Java 面试里的“必考题”，很多初学者都会被绕晕。来，老师给你打个最通俗的比方，保证你一听就懂！")
	# ......
]

# 将历史信息注入到聊天模板中去
prompt = chat_template.invoke({"history":history_data})
```

​	将其变为chain链接的话，就是如下操作：
```py
# 这里需要注意区分哦，在调用chain的时候，不能直接使用prompt传给model，因为prompt已经是incoke后的结果了
prompt = chat_template.invoke({"history":history_data})
print(prompt)

# 这里就是chain的部分了，将其组成一条链路
chain = chat_template | model
print(chain.invoke({"history":history_data}))
```



###  补充：

#### .format() 和 .invoke() 的区别

​	两者都是由继承 BasePromptTemplate 得到的，但前者功能是单纯的用字符串替换，直接返回字符串，而后者是 Runnable 接口的标准方法，返回的是 PromptValue 的类对象，而且后者的传参需要传入字典，将 kv 对应，而前者不需要。在使用 Runnable 接口的时候，最好使用后者 .invoke() 方法
​	所以如果需要纯字符串的返回，就用 .format()，而需要使用 chain 的话，就需要用 .invoke()

#### Prompt 模板接收对象类型

​	Prompt 模板主要有两种核心调用方式，依据不同的调用方式，传参的格式也不同
​	使用 .invoke() 方法时，必须传入的对象类型是**字典**。这是因为 .invoke() 是 LangChain 标准化运行接口（Runnable）的一部分，它需要统一接收一个完整的输入对象
​	而使用 .format() 或 .format_messages() 方法时，则是直接传入变量。因为传统的 .format() 方法它的用法和 Python 原生的字符串格式化非常像，**不需要字典**，而是直接将变量作为关键字参数进行传入即可
​	除了这两种情况，其实还有第三种情况，就是自己在实例化对象的时候可以直接支持多种格式的输入，比如：

```py
from langchain_core.prompts import PromptTemplate
# 第三种情况：使用 partial_variables 在实例化阶段进行固定部分变量
prompt = PromptTemplate(
    template="系统角色：{system_role}\n当前时间：{date}\n用户问题：{question}",
    input_variables=["question"],           # 运行时动态传入的变量
    partial_variables={                     # 实例化时直接固化（支持硬编码/环境变量/函数返回值）
        "system_role": "你是一个资深Java架构师",
        "date": "2024-05-20"
    }
)

# 调用时框架会自动合并 partial_variables，只需传入缺失的 key
res = prompt.invoke({"question": "Spring Boot 如何实现分布式锁？"})
print(res.text)
```



## Chain 链

​	LangChain 中的 Chain 就是其核心，而其说白了就是将多个组件像链条一样拼凑起来，形成一个可以被复用的工作流，其核心设计理念就是：**前一个步骤的输出，会自动成为下一个步骤的输入**
​	在我们之前的针对提示词模板那一块部分中，就用到了 Chain ，我们将提示词模板与大语言模型链接起来，将前者的输出传入后者，形成了一个最基础的工作流，在这之中我们可以发现两个组件之间使用的是“ | ”（或运算）链接，意思是将前者输出传入后者
​	组成链也是有核心要求的，就是组成 Chain 链的组件必须是 Runnable 接口的子类对象，虽然绝大多数组件都是其对象，但仍然可能存在搞混淆的情况。比如之前在 ChatPromptTemplate 部分中提到的 __“这里需要注意区分哦，在调用 chain 的时候，不能直接使用 prompt 传给 model ，因为 prompt 已经是 incoke 后的结果了”__，这里就是表明，其 prompt 已经是一个结果了，是一个字符串的结果，而不是原来的类对象了，所以不可以传给 model 这个组件

```py
# 可以使用 type() 方法检测 chain 的类型，会发现其类型是<class 'langchain_core.runnables.base.RunnableSequence'>
# 在 chain 后面加不加这个 RunnableSerializable 其实都一样
chain= chat_template | model
print(type(chain))
```

​	chian 的流式输出和非流式输出和模型直接调用流式非流式一样的操作方式
### 补充：

#### Chain 的嵌套

​	就是已经创建的 Chain 可以嵌套进下一个 Chain 中，只要符合以上的要求就行，大大的提高了可用性

#### Chain 的输入

​	Chain 本质就是一个组件的输入处理后将输出传给下一个组件，使用使用 chain.invoke() 的时候，要注意根据第一个组件的输入来写



## Runnable接口

​	LangChain 中的绝大多数组件都是继承自 Runnable 抽象基类，而 LangChain 则是对这个基类中的“ | ”（或运算__ or __）进行了重写，返回得到 RunnableSequence 对象，就是上面说的Chain的类型
```py
# 可以点进去看看其继承的接口情况，最终都会指向Runnable接口
from langchain_core.prompts import PromptTemplate
```



## StrOutputParser字符串输出解析器

​	StrOutputParser（太长了往后简称：SOP） 是 LangChain 内置的简单字符串解析器，是其中最基础也是最常用的解析器，作用是将 LLM 返回的复杂的消息对象（比如 AIMessage ）转变为最普通的Python的字符串（str），并且其也是属于 Runnable 抽象基类的子类，可以作为组件放在 Chain 中去
​	在使用它之前，我们按照**常理**来看构建 Chain 可以是这样的：

```py
# 将 prompt 传给LLM，然后LLM的返回又再一次传给LLM
chain = prompt | model | model
```

​	按照常理来看确实没错，但是运行时会报错
​	因为在执行第一段链的时候，prompt 传给 model 后，model 返回的数据类型不再是 Chain 可以接收的 Runnable 基类的子类，而是比较复杂的消息对象，这种消息对象就不再可以直接传给下一个 model
​	而有了 SOP 就不一样了，它可以将这种复杂对象转变为 LLM 可以识别的 str（字符串），继续传给下一个 model：

```py
# 将 prompt 传给 model，然后 model 的返回传给解析器解析，解析完成后返回一个 model 在 chain 中可以接收的数据，然后又再一次传给下一个 model（其实也可以是同一个，就是上下游层级不同，看情况业务而定）
parser = StrOutputParser()
chain = prompt | model | parser | model
```

​	SOP 因为其简便性使得其成为最常用的解析器，不像其他解析器一样需要传入复杂的格式指令或 Schema，其可以直接通过上面的方式进行实例化后直接使用。并且 SOP 是可以支持流式输出的，将 model 吐出的一个个文本片段（chunk）实时转换成字符串
​	以下是实际使用的例子：

```py
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatTongyi(
    model="qwen3-max",
    api_key=os.getenv("TONGYI_API_KEY")
)

prompt = PromptTemplate.from_template(
    "我的邻居姓{lastname}，刚生了{gender}，你来帮我起一个名字，回答简单"
)
# 上面的和以往一样，主要是下面的部分
parser = StrOutputParser()
chain = prompt | model | parser | model
res = chain.invoke({"lastname":"张","gender":"女孩"})

print(res.content)
print(type(res))
```

​	这个例子中可以完美表达 SOP 的作用，但是存在一个问题，就是在例子中，下游的 model 在接收到上游的 model 传过来的名字后，不知道这是要干什么，回答的很懵逼，而实际操作中我们应该是得到了上游 model 传来的名字后，进行数据的处理，比如提示词填写，然后将名字与提示词一起传给下游的 model ，为了实现真实情况，就要引入下一个 JsonOutputParser 解析器



## JsonOutputParser JSON字符串输出解析器

​	JsonOutputParser （还是太长了，往后简写为：JOP）这也是 LangChain 中非常常用的解析器，主要功能就是强制将 model 的输出变成标准的 JSON 格式，并自动将其转换为 Python 中的字典或列表
​	从直觉上来看，加入这个解析器后我们的实现链路是不是这样？

```py
# 实例化解析器
s_parser = StrOutputParser()
j_parser = JsonOutputParser()

f_prompt = PromptTemplate.from_template(
    "我的邻居姓{lastname}，刚生了{gender}，你来帮我起一个名字，回答简单"
)
s_prompt = PromptTemplate.from_template(
    "姓名:{name}，请帮我解析含义"
)
# 构建链路
chain = f_prompt | model | j_parser | s_prompt | model | s_parser
res = chain.invoke({"lastname":"张","gender":"女孩"})

print(res)
print(type(res))
```

​	但是实际运行起来会报错，为什么呢？
​	因为 json 解析器是将输出转化为 json 的字典，在这里前面的 model 返回的仅一个名字、一个字符串，而json解析器无法将字符串转变为一个字典，因为这个东西它没有key-value，所以会执行失败
​	为什么实现 k-v 的对应，就需要从提示词开始着手，将返回的内容变为字典

```py
# 只需要将提示词改成这样就可以跑通了
f_prompt = PromptTemplate.from_template(
    "我的邻居姓{lastname}，刚生了{gender}，你来帮我起一个名字，回答简单,并封装为json格式返回，key就是name，value就是你起的名字，严格按照格式要求返回"
)
```



## RunnableLambda

​	这是 LangChain 中极其灵活的一个组件，作用十分直观灵活，就是将你写的任意一个普通 Python 函数或者 lambda 匿名函数，包装成 LangChain 标准的 Runnable 组件，让其可以插入 chain 中去
​	在上一个 json 解析器中，其必须对提示词进行修改，才能确保返回的字符串可以转变为 json 格式的字典，而利用 RunnableLambda 让我们可以自己定义一个函数来实现不需要修改提示词就可以实现转换

```py
# 前面都差不多，这里添加一个新依赖
from langchain_core.runnables import RunnableLambda

f_prompt = PromptTemplate.from_template(
    "我的邻居姓{lastname}，刚生了{gender}，你来帮我起一个名字，回答简单不要说别的"
)
s_prompt = PromptTemplate.from_template(
    "姓名:{name}，请帮我解析含义"
)
# 解析器的实例化
s_parser = StrOutputParser()
# 重点在这，构建一个自定义函数
my_func = RunnableLambda(lambda ai_msg: {"name":ai_msg.content})

chain = f_prompt | model | my_func | s_prompt | model | s_parser

for chunk in chain.stream({"lastname":"王","gender":"男孩"}):
    print(chunk,end="",flush=True)
```

​	通过这种方式，我们可以直接将返回的名字变为 json 的字典传给下一个 model
​	其实用这种方式也可以不用实例化

```py
# 可以直接写进去，不用实例化
chain = f_prompt | model | (lambda ai_msg: {"name":ai_msg.content}) | s_prompt | model | s_parser
```

​	当然实例化是为了更好的实现复用和可用性，满足单一职责原则和接口隔离原则



## Memory 记忆

​	LangChain 中的 Memory 是让 LLM 拥有“记性”的核心组件，因为 LLM 本身是无状态的，每一次对话就是一次全新的开始，而 Memory的作用就是将用户与 LLM 的对话内容进行缓存，在每一次对话中自动的将历史对话拼接到 Prompot 中去。而 LangChain 中的记忆方式有两种，**临时记忆**和**长期记忆**

### 临时记忆

​	也可以叫做短期记忆，核心作用就是让 LLM 在**当前**这一轮的对话中保持记忆的连贯，而在程序结束后，记忆就会丢失，LLM 会彻底失忆重新开始
​	其中主要包含三个组件：InMemoryChatMessageHistory，MessagesPlaceholder，RunnableWithMessageHistory：
​	**InMemoryChatMessageHistory** 内存存储本，这是 LangChain 提供的一个最基础的存储类，它的作用就是在 Python 程序的内存（字典）里，临时保存用户和 AI 的对话消息列表
​	**MessagesPlaceholder** 提示词占位符，它相当于在你的System Prompt里提前挖好一个“坑”，每次调用 LLM 时，LangChain 会自动把历史聊天记录填进这个坑里，让LLM能看到之前的对话上下文
​	**RunnableWithMessageHistory** 记忆包装器，这是 LCEL（链式调用）中的“管家”，它负责在每次你提问前，自动去内存里把历史记录捞出来，等 LLM 回答完后，又自动把新的对话存回内存里

```py
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

# 这里有两种提示词模板的方式，第一种是通用提示词，第二种是聊天提示词，整体用下来第二种更好一些，所以将第一种注释了

# prompt = PromptTemplate.from_template(
#     "你需要根据历史会话记录来回应用户问题。对话历史：{chat_history}，用户提问：{input}，请回答"
# )
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","你需要根据历史会话记录来回应用户问题。对话历史："),
        # 这个组件在之前聊天提示词模板的地方用过，用于加载会话历史
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
        # 这个组件虽然只有一行，但是在每一用户进行访问的时候，都会创建一个
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
```

### 长期记忆

​	长期记忆与短期记忆的最大区别就是他可以让 LLM 跨越会话跨越时间，仍然可以记住用户的重要信息

#### FileChatMessageHistory 
​	LangChain 本身是只提供一个内存存储的功能，也就是短期记忆，当程序运行结束的时候，记忆就会被自动删除，为了永久保存记忆，我们可以自定义一个类 FileChatMessageHistory ，这个类继承自 BaseChatMessageHistory（其中的 AIMessage 、HumanMessage 、SystemMessage 都是 BaseMassage 的子类），用这个类来实现长期存储，将历史记录保存起来，我们同样根据上面的短期记忆的id查找来编写
​	这个类需要定义入口函数，接收会话ID session_id 和仓库路径 storage_path，并将其变为全局成员变量，同时拼凑为完整文件路径：

```py
# 继承BaseChatMessageHistory
class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self,session_id,storage_path):
        self.session_id = session_id    # 会话的id
        self.storage_path = storage_path    # 不同会话id的存储文件所在的文件夹路径
        # 拼接为完整的文件路径，调用os中的路径拼接方法
        self.file_path = os.path.join(self.storage_path,self.session_id)
        # 首先确保文件所在的文件夹路径存在，如果不存在就创建，存在就跳过，确保不会报错
        os.makedirs(os.path.dirname(self.file_path),exist_ok=True)
```

​	除了入口函数外，还需要实现父类的三种核心方法：add_message、messages、clear
​	首先看 add_message ，他的作用是当产生新的对话时，先把文件里原有的记录读取出来，加上新对话，然后打包写入文件。因为电脑文件不能直接存 Python 的消息对象，所以用了 message_to_dict 把消息转换成字典，再用 json.dump 变成文本存进文件

```py
	# 可以将Sequence序列当作是list这种，而BaseMessage则是AIMessage等message的父类
    def add_message(self,message: Sequence[BaseMessage]) -> None:
        all_messages = list(self.messages)
        all_messages.extend(message)

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
```

​	再就是编写 messages 函数，这个函数编写完成后需要将其转变为一个属性方法，使用@property 装饰器实现。而当每次 AI 需要回忆时，就会调用它。它会去对应的文件里读取 JSON 文本，并通过 messages_from_dict 把字典还原成 AI 能看懂的消息对象列表。如果文件还没创建（FileNotFoundError），就返回一个空列表。
```py
    @property # @propery装饰器，将messages方法变为成员变量属性
    def messages(self) -> list[BaseMessage]:
        # 使用try防止找不到报错
        try:
            with open(self.file_path,"r",encoding="utf-8") as f:
                messages = json.load(f)
                return messages_from_dict(messages)
        except FileNotFoundError:
            return []
```

​	最后需要实现 clear 方法
```py
    def clear(self) -> None:
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump([],f)
```

​	完成这一切后，FileChatMessageHistory 类算是编写成功，可以实现长期存储功能，调用它也十分方便，可以直接在短期存储的代码中来实现，只需要将get_message方法进行简单的修改
```py
def get_history(session_id):
    # 这里的意思是在当前路径下的 chat_history 文件夹中查找，没有会自动创建
    return FileChatMessageHistory(session_id,"./chat_history")
```

​	但在实际使用的情况中，我发现系统会这样一个错误
```bash
Error in RootListenersTracer.on_chain_end callback: AttributeError("'tuple' object has no attribute 'type'")
```

​	可能是因为版本的兼容问题，使得其表示：**代码的某个地方正在尝试获取一个“元组（tuple）”的 `type` 属性，但元组这种数据类型本身并没有 `type` 这个属性**。解决的话就需要修改 add_message 方法
```py
    # 修改了 message 的类型
    def add_message(self,message: BaseMessage) -> None:
        all_messages = list(self.messages)
        # 以及添加的方式
        all_messages.append(message)
        new_messages = [message_to_dict(message) for message in all_messages]
        
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump(new_messages,f)
```

​	随后就可以实现功能长期存储的功能，而且将文件保存在当前路径下的 chat_history 文件夹的 user001 文件中
​	往后的长期记忆还是有很多部分的，但正在深入就是企业级架构的层面了，单单一个学习笔记应该是无法讲透的，主要是我自己也还没学太明白更深入的东西



## Document Loaders 文档加载器

​	LangChain 中的 Document Loaders 是构建RAG应用的第一步，其提供了一套保准接口（ Document Loaders 一般都需要实现 BaseLoader 接口），作用就是将各种各样的外部数据（比如PDF、网页、Word等）统一转换为 LangChain 可以识别的 Document 对象（调用方法返回 Class Document 的实例）
​	而这个 Document 对象主要就包括一个字符串（page_content）和一个字典（metdata），字符串就是文档的实际内容，而字典则是文档的元数据，比如页码、作者、时间等
​	Document Loaders 面对不同的数据包含有不同的加载器，但无论哪一个加载器，都将实现统一的接口方法 .load( ) 和 .lazy_load( )。其中前者表示一次性的加载所有文档，而后者则是流式的传输文档
​	LangChain 中内置超级多的文档加载器，而我们只需要了解几个比较常用的就行

### CSVLoader

​	LangChain 中专门用来处理 CSV 表格文件的文档加载器，其核心逻辑就是默认将 CSV 表格中的每一行数据，转换成一个独立的 Document 对象
​	以下是简单使用

```py
from langchain_community.document_loaders import CSVLoader

lodaer = CSVLoader(
	# 需要创建文件，在当前路径下的CSV文件夹中创建
    file_path="./CSV/user.csv"
)

documents = lodaer.load()
# 打印查看类型与内容
for document in documents:
    print(type(document),document)
```

​	一般来说这样子就可以直接运行得到结果，但是 windows 用户会报错，因为其默认编码方式是 gbk 编码，这中编码会出错，所以一般要在方法中指定编码方式为 utf-8
```py
lodaer = CSVLoader(
    file_path="./CSV/user.csv",
    encoding = "utf-8"
)
```

​	而 .lazy_load() 的加载方式呢，其实就是用 for 循环加载，使用该方法会返回一个迭代器（和流式输出很像吧）
```py
documents = lodaer.lazy_load()
for document in documents:
    print(type(document),document)
```

​	CSVLoader 可以自动识别 csv 文件以什么为分隔符，默认是使用 “，” 为分隔符，也可以进行修改，只需要在调用的时候指定参数就行：
```py
lodaer = CSVLoader(
    file_path="./CSV/user.csv",
    # 里面放的是字典
    csv_args={
    	# 可以将这里的值换成任意一个
        "delimiter":","
    },
    encoding = "utf-8"
)
```

​	而这个 csv_args 参数不只可以这样，当我们面临 csv 文件中的属性有使用逗号或句号这种可能被识别为分隔符的时候（比如 李四，30，上海，“吃饭，睡觉”），就可以对着个参数进行修改
```py
lodaer = CSVLoader(
    file_path="./CSV/user.csv",
    csv_args={
        "delimiter":"|",
		# 指定被‘ " ’包裹的不进行分割
        "qutotechar":' " '
    },
    encoding = "utf-8"
)
```

​	当数据本身并没有带表头的时候（就是没有姓名,年龄,城市），可以在其中修改 fieldnames 这个参数
```py
csv_args={
        "delimiter":"|",
        "quotechar":'"',
    	# 表示添加表头a,b,c
        "fieldnames":["a","b","c"]
    },
```



### JSONLoader

​	JSONLoader 是 LangChain 中专门用于处理 JSON 数据的加载器，其功能主要是将 JSON 或 JSONL 格式的结构化数据，转换成 LangChain 能够识别和处理的 Document 对象列表
​	使用这个加载器之前需要从社区库下载 jq ，这是一种专门用于处理JSON数据的语言，下载很简单，直接在终端输入：

```py
pip install jq
# 或 uv 环境安装（conda也差不多，注意甄别运行环境）
uv pip install jq
```

​	JSONLoader 的实例化中，有两个参数必填的，就是 file_path 和 jq_scheme ，前者表示抽取的文件路径，后者表示抽取些什么
```py
loader = JSONLoader(
    file_path="./CSV/user.json",
    # 只有一个 . 表示全部都抽取
    jq_schema="."
)
```

​	但是按照这个方式直接输出的话，是会报错的，这一次报错不再是编译器错误，而是出现值错误。什么意思呢？就是这找出来的 page_content 他只能是字符串，不可以是字典， json 格式找出来的整体在 python 中是一个字典类型，所以会报错。而为了解决报错，就需要在其中指明，我们抽取的目标不是字符串，就要使用到 text_content
```py
loader = JSONLoader(
    file_path="./CSV/user.json",
    jq_schema=".",
    # 这个参数默认是True，表示查找的是字符串，改成False就是不是字符串类型
    text_content=False
)
```

​	json 文件可以不只是一个对象，可以是多个对象放在一起，比如一个 json 数组
```json
[
  {
    "city": "北京",
    "weather": "晴",
    "temperature": 22
  },
  {
    "city": "上海",
    "weather": "小雨",
    "temperature": 18
  },
  {
    "city": "广州",
    "weather": "多云",
    "temperature": 26
  }
]
```

​	从这种类型的 json 中抽取，可以改变 jq_schema 的抽取对象：
```py
 loader = JSONLoader(
    # 注意这里换文件了
    file_path="./CSV/weather.json",
    # 抽取所有的数组的每一个元素的city属性
    jq_schema=".[].city",
    text_content=False
)
```

​	还有一种 json 文件，它对于标准的 json 语法来说是非法的，不是一个标准的 json 文件
```json
{"id": 1, "name": "Alice", "active": true}
{"id": 2, "name": "Bob", "active": false}
{"id": 3, "name": "Charlie", "active": true}
{"id": 4, "name": "Diana", "active": false}
```

​	但是其每一行的每一个对象都是合法的，放在一起就不合法了，这种情况的话用之前抽取数组的方式也无法实现，因为其不是数组。而抽取这种文件的话，还需要用到一个参数 json_lines 
```py
loader = JSONLoader(
	# 又换文件了哦
    file_path="./CSV/user_json_lines.json",
    jq_schema=".name",
    text_content=False,
    # 同样的，这个参数默认是False，表示这不是json_line的文件，改为True就让其知道这是json_line
    json_lines=True
)
```

​	JSONLoader 中只有两个参数是必要参数 file_path 和 jq_schema，而其他的都是非必要参数



### PyPDFLoader

​	是 LangChain 中关于提取 PDF 文件的加载器，LangChain 中有很多 PDF 加载器，而 PyPDFLoader 则是其中比较轻量、速度快，也是 LangChain 中处理常规 PDF 的首选。
​	使用加载器之前需要下载安装它的依赖库 pypdf ，直接在文件下的终端中下载：

```bash
pip install pypdf
# 或者uv环境下
uv pip install pypdf
```

​	该加载器仅一个参数必填，还是 file_path ，但实际使用下来，还是下面这种方式更加方便快速

```py
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    file_path="./data/document.pdf",    # 文件所在路径
    mode='page',       # 采用哪一钟读取模式，有page（按页面划分document）和single（只生成单个的document），默认page
    password='password'     # 读取文件需要的密码是什么
)
```

​	虽然不适用下面两个参数也可以得到结果，但是使用后会更加可控



### TextLoader

​	这是 LangChain 中最直观、最基础的文档加载器了，主要作用就是读取本地或远程的纯文本文件（如 `.txt`、`.md` 等），并将其内容**全部**封装成 LangChain 能够统一处理的 Document 对象
​	它的使用只需要填写一个必要参数 file_path 

```py
loader = TextLoader(
    file_path="./data/dately.txt",
    encoding="utf-8"
)
```

​	这里添加 encoding 参数，原因和之前一样，就是单纯的 windows 编译问题，需要指明编译方式。但是在实际使用中还是容易出现问题，比如说乱码，而为了防止出现乱码，就需要另一个参数 autodetect_encoding
```py
loader = TextLoader(
    file_path="./data/dately.txt", 
    encoding="utf-8", 
    autodetect_encoding=True
)
```

​	这个参数的核心作用就是在 TextLoader 读取文件之前先自动扫描一遍文件的字符编码，然后猜测的用正确的编码去识别读取
​	这个加载器的内容呢十分简单，有这些基本就差不多了，但是这个加载器，有一个缺点，就是他不会直接将文本进行分割。什么意思呢？就是他返回的 document 对象，只有一个，如果你使用 .len() 方法去包裹观察输出的内容的话，你会发现它只会返回1，那如果这个文本很大怎么办？只返回一个是不是很影响性能？
​	所以这时候要用到 TextSplitter 文本分割器（递归字符文本分割器 RecursiveCharacterTextSplitter）

```py
# 注意这里的导包和以往不同
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,		# 分段的最大字符
    chunk_overlap=50,	# 分段之间可允许的最大重叠字符数目
    # 文本分割的自然段落依据符号
    separators=["\n\n","\n",".","?","!","。","？","！"," ",""],
    length_function=len	# 用来计算字符数量的函数
)
```

​	使用这个方法也很简单，调用其中的 .split_documents() 函数就行了
```py
res = loader.load()
split_docs = splitter.split_documents(res)
```

​	然后它的输出因为分割器将其分成了多个 document ，所以变成了迭代器，需要循环输出
```py
# 测一下长度
print(len(split_docs))
for doc in split_docs:
    print("="*20)
    print(doc)
    print("="*20)
```



## Vector stores 向量存储

​	在 LangChain 中，向量存储是构建RAG的关键，其主要作用是将文本转化为机器能理解的数字向量并存起来，在需要的时候，通过语义快速找出最相似的内容（相似度）
​	向量存储的工作流主要分为两个阶段，也是对应于RAG的两个阶段：
​	**索引阶段**：把一堆文档（比如公司制度、产品手册）交给 LangChain 时，会先调用“嵌入模型 Embedding Model ”把每一段文本转换成一个高维的数字向量（一串数字）。然后，向量存储会把这些向量连同原始文本一起保存进数据库。
​	**查询阶段**：当向 AI 提问时，系统会将问的问题也转换成向量，然后向量存储会在数据库里进行“相似性搜索”，找出和问题的向量在数学空间上距离最近的那些文档片段，最后把这些片段喂给LLM去生成答案
​	索引阶段的“将文本转换为向量”的这一部分，在之前了解 embedding 模型的时候就已经使用过的，而现在就是将转换的向量储存起来，并实现查询阶段的功能

### 内存向量存储

​	这是向量数据库中的非常轻量级的存储形式，主要就是将生成的文本向量数据直接存储在本地的内存中，而不是写入数据库，使得其读取和检索的速度极快，并且不需要额外安装数据库依赖。但缺点也很明显，检索的结果并不精确，且不能长期存储，同时无法处理高维度的向量数据
​	而它的使用则是需要与文本嵌入模型配合：

```py
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
# 这个就是内存向量存储实现
vector_store = InMemoryVectorStore(
    embedding=DashScopeEmbeddings(
        # 因为我没有把API_KEY保存到环境变量，所以使用的时候需要表明嵌入模型的key
        # 这里省略了从.env中获取api_key的步骤，因为之前有很多，可以回头看看
        dashscope_api_key=os.getenv("TONGYI_API_KEY")
    )
)
```

​	完成实现后还需要将需要处理的文本加载进来，采用之前学的 CSVLoader 进行加载
```py
# 包就不展示导入了
loader = CSVLoader(
    file_path="./data/info.csv",
    encoding="utf-8",
)
documents = loader.load()
```

​	针对内存向量存储，主要还是 CRUD 的操作:
​	新增（creat）

```py
vector_store.add_documents(
    # 导入需要添加的文档列表
    documents = documents,
    # 这里是给每一个文档都添加id，比如第一个文档标签就是id1
    ids=["id" + str(i) for i in range(1,len(documents)+1)]
)
```

​	删除（remove）
```py
vector_store.delete(
    # 这里表示删除列表中id为多少的文档
    ["id1","id2"]
)
# 注意啊，删除之后可能会导致查找不到想要的
```

​	查询（query）
```py
# 查询是返回列表的，所以需要赋值
res = vector_store.similarity_search(
    "衣服怎么样",	 # 这是查询的语句
    1				# 这是查询几条文档出来
)
```

​	你可能要问：“不是说 CRUD 嘛？修改操作去哪里了？”，啊其实 LangChain 的这个向量存储的抽象层没有原生的修改操作，因为向量这个东西它具有不可变性，修改一个向量不现实，只能全部让嵌入模型重新进行计算。而且向量库一般采用 **最终一致性 + 追加写入** 的模型，避免传统数据库的行级锁与事务开销，以换取极高的检索吞吐，所以其本身是不支持修改的。所以针对修改，也只能曲线进行，也就是**先删后增**的模式实现



### 外部向量存储

​	说完了内存向量存储，再来看外部向量存储。首先我们知道内部向量存储并不持久，且出现故障就容易暴毙。而在 LangChain 中，外部向量存储就可以很好的解决它的缺点
​	外部向量存储指的是那些能够将向量数据**持久化保存**在磁盘、本地数据库或云端服务器上的存储系统。无论程序重启多少次，或者服务器发生宕机，只要再次连接到同一个存储地址，之前存入的向量数据依然完好无损
​	我们之前学过 Chat Model 的 Memory ，其中也包含了短期记忆和长期记忆，这两个概念在文本向量中也是刚好对应内存存储和外部存储的。外部向量存储是现在构建任何**生产级 RAG**（检索增强生成）应用必不可少的基础设施
​	需要实现外部向量存储，就需要引入向量数据库。LangChain 中可以使用轻量级的 Chroma 向量数据库

```bash
# 使用前需要确保安装了对应依赖库
pip install langchain-chroma chromadn
# 或uv
uv pip install langchain-chroma chromadn
```

​	确定安装后，就需要导入 Chroma 向量数据库了
```py
from langchain_chroma import Chroma
# 这个数据库类似于Python的SQLite
vector_store = Chroma (
    collection_name="test",						# 这里表示设置创建的数据库的名称，相当于表名
    embedding_function=DashScopeEmbeddings(		  # 这里和内存存储是一样的，确定嵌入模型是什么
        dashscope_api_key=os.getenv("TONGYI_API_KEY")
    ),
    persist_directory="./data"					 # 这里就是指定数据库存储的位置 
)
```

​	除此将内存向量存储的实现改为 Chroma 向量数据库的实现外，其他代码与之并无大差异
​	而第一次进行了添加操作之后，我们就可以发现当前目录下的data目录多出来两个文件，就是向量数据库的文件
​	完成第一次的初始添加操作后，将添加语句删除再试，会发现其可以直接回答

```py
loader = CSVLoader(
    file_path="./data/info.csv",
    encoding="utf-8",
)
documents = loader.load()
# 顺便把删除操作也注释了
# vector_store.add_documents(
#     documents = documents,
#     ids=["id" + str(i) for i in range(1,len(documents)+1)]
# )
# 
# vector_store.delete(
#     ["id1","id2"]
# )

res = vector_store.similarity_search(
    "衣服怎么样",
    1,
)

print(res)
```

​	再就是关于查询操作，当数据量很多的时候也可以设置查询条件进行过滤
```py
res = vector_store.similarity_search(
    "衣服怎么样",
    1,
    filter={"category": "语义标签"}	# 左边是key右边是value，可以表示根据这个key，找和value一样的文档
)
```

​	现在已经学会了创建并检索向量库，也可以将其与 Chain结合起来使用。但是这种使用并不是将检索这一过程入链，只是单纯的把它当作一个函数使用，用函数的结果调用链条，这在实际开发中非常不合理
​	为了实现将向量检索加入Chain中去，就需要用到 RunnablePassthrough



## RunnablePassthrough

​	是 LangChain 中用于构建复杂链式调用时的“数据搬运工”，核心逻辑很简单，就是输入是什么，输出就是什么，它不做任何计算、不改变数据内容，只是单纯地把数据“传递”下去。而这个时候你可能会有疑惑，不是说入链嘛？用这个单纯的传输数据的东西能干嘛？
​	首先我们要明白为什么向量数据库没办法入链

```py
vector_stores = InMemoryVectorStore(
    embedding=DashScopeEmbeddings(
    	# 这里换了一个嵌入模型，这个会比较牛逼
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
```

​	这里我创建了一个内存向量存储的短期向量数据库 vector_stores ，而从 InMemoryVectorStore 的源码中可以知道，它是不属于Runnable 接口的子类的。也就是说它的实例化 vector_stores 没办法加入Chain链中去
​	但是哈，LangChain 为每一个向量数据库都提供了一个继承组件 .as_retriever() ，这个方法呢可以返回一个 Runnable 的子类实例对象，就可以解决向量数据库实例无法接入 Chain 的情况了
​	完成了转换就可以将 vector_stores 转变为一个符合 Runnable 接口的实例 retriever

```py
retriever = vector_stores.as_retriever(
    search_type = "similarity", # 表示用什么方式搜索，这里是相似度检索
    search_kwargs = {"k" : 2}   # 表示只要最相近的两条结果
)
```

​	这里可以发现和前面学的检索向量库的内容很相近
​	但是这和 RunnablePassthrough 有什么关系？都已经完成转换了，直接加入 Chain 不就行了吗？

```py
# 按直觉来看，是不是可以直接这样？
chain = retriever | prompt | model | strparser()
```

​	很遗憾这是无法实现的
​	前面在学习提示词模板的时候就有补充，这个链里面的 prompt 接收的输入是什么？可以是字符串也可以是字典。那 retriever 输出的是什么？是向量库检索结果是一个文档列表（ list[Document] ），一个列表如何被 prompt 接收？
​	一个提问可以给 retriever ，因为提问是字符串，但是 retriever 的结果给不了 prompt ，这个 chain 仍然无法实现
​	我们可能会想起来，不是可以自己写一个函数来实现输入输出的类型转换吗？
​	可以，但是 prompt 要的输入可不只是检索的结果，重点在于用户的提问。而用户的提问在最开始就传给了 retriever ，就已经没有了，就算实现类型转换，也无法做到完整的完成Chain
​	这时候就需要这个RunnablePassthrough了

```py
chain = (
    # 这个东西就像一个占位符，可以接收输入，可以截流
    # 这format_func也是自定义函数，作用是收到的list组合成一个字符串
    {"input" : RunnablePassthrough(),"context" : retriever | format_func }
    | prompt
    | print_prompt	# 自定义函数，用于观看最终的prompt是什么
    | model
    | strparser
)
```

​	RunnablePassthrough() 的作用是什么？刚刚已经说了，就是输入什么就输出什么。而这个特性刚好可以完美的解决 prompt 缺失用户提问的问题
​	在上面这个chain中，可以发现chain的起始组件变为了一个字典，事实上字典也是满足传入的，其父类中的Mapping属于LangChain中Chain定义的组件类型
​	在这里面 RunnablePassthrough 的作用就是做一个占位符接收用户的问题输入，将这个输入截流下来，原始输入就传给了 retriever ，而它截流的输入和 retriever 的输出一起组成了一个字典传给 prompt ，刚好完美解决！
​	完整代码如下：

```py
import os
from xml.dom.minidom import Document
from dotenv import load_dotenv
from huggingface_hub import search_spaces
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.output_parsers import StrOutputParser
# 关键导入
from langchain_core.runnables import RunnablePassthrough
# 模型调用
load_dotenv()
model = ChatTongyi(
    model="qwen3-max",
    api_key=os.getenv("TONGYI_API_KEY")
)
# 提示词模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "请严格基于我提供的参考资料（{context}）来回答，不要编造信息。如果资料里没有答案，请直接说‘资料不足’。回答要简洁、有逻辑，引用资料时请标注关键信息。"),
        ("user", "问题：{input}")
    ]
)
# 向量数据库的创建
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

# 自定义两个函数
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
# chain的构建
chain = (
    {"input" : RunnablePassthrough(),"context" : retriever | format_func}
    | prompt
    | print_prompt
    | model
    | strparser
)
# 这里使用的 invoke 进行输入
res = chain.invoke(input_text)

print(res)
```

**总结一下**：
	使用 LangChain 构建完整的 RAG应用，核心在于打通数据的流转。
	底层的向量数据库本身只是存储向量数据的容器，必须通过 LangChain 提供的 as_retriever() 方法将其适配为标准的检索器，使其具备接入 Chain 链式调用的能力。
	然而，如果直接将检索器接入链条，不仅输出的文档列表格式无法被提示词直接识别，还会导致用户的提问在检索过程中丢失
	而LangChain提供的 RunnablePassthrough() 这一“数据搬运工”与字典映射机制就完美的解决了这一痛点，使的可以构建并行的数据分流：
	一条流向检索器获取资料并经由自定义函数格式化为字符串，另一条通过 RunnablePassthrough() 完美保留用户的原始提问，流向其他
	这两条数据流汇聚成包含完整上下文和用户问题的字典，精准地喂给 prompt 与LLM，从而实现了逻辑严密、数据完整的自动化问答链


# 总结

​	走过这一程，我们从最基础的模型调用起手，一步步拼出了 Prompt 模板、Chain 链、记忆模块、文档加载与向量检索，最终打通了完整的 RAG 问答链路。你会发现，LangChain 的强大从来不是某个单点 API，而是它用 **Runnable 协议**将一切组件拉平到同一张调度网上。

​	前一步的输出自动成为下一步的输入，数据在 Prompt、Model、Parser、Retriever 之间严丝合缝地流转。

​	掌握它的关键，不在于背熟参数，而在于建立**“数据流视角”**：时刻清楚当前环节吐出的是什么类型（ AIMessage？list[Document]？dict？），下游接住的是否匹配。遇到报错，先查类型契约，再看链路由；写好功能，再想边界处理与持久化。从内存短期记忆到文件长期存储，从直觉式函数调用到 RunnablePassthrough 的并行分流。

​	希望我的这份学习笔记可以帮到你，为你铺好一块基石。大模型的相关技术日新月异，但其中的工程哲学不会过时。有道无术尚可求，有术无道止于术。希望我们能在这个时代找到自己的定位，实现自己的价值！	
