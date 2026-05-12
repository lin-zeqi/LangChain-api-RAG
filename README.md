# 📘 LangChain 实战笔记与配套代码

> 以「数据流视角」拆解 LCEL 链式架构，从原型跑通到 RAG 落地实践
> ps:具体笔记文档超链接在最下面！！

本仓库系统整理了本人对于 LangChain 核心组件的学习笔记与完整可运行代码示例。
内容以 **「组件解耦 + 链式组装」** 为主线，覆盖模型调用、提示词工程、Chain 工作流、记忆管理、文档加载、向量检索至完整 RAG 链路搭建。所有代码均基于 Python 与阿里云百炼（DashScope）后端验证，适合作为入门指南、调试参考与工程实践模板。

---

## ✨ 核心亮点

| 特性 | 说明 |
|------|------|
| 🔍 **数据流驱动** | 强调“上游吐出什么类型，下游能否接住”，提供完整的类型流转对照表与契约对齐指南 |
| 🛠️ **工程避坑实录** | 收录实战高频报错与修复方案（如 `tuple` 类型兼容、`format()` 破坏链式协议、向量库无原生 `update` 等） |
| ⚡ **开箱即用** | 每个模块均配备独立可运行脚本，目录结构清晰，依赖集中管理，替换 Key 即可跑通 |
| ☁️ **百炼深度适配** | 示例默认对接 `ChatTongyi` 与 `DashScopeEmbeddings`，兼容 `qwen-max` / `qwen3-max` / `text-embedding-v4` 等模型 |
| 📐 **生产思维前置** | 在原型代码中预埋环境变量加载、流式输出控制、元数据过滤与 Token 监控等工程化注释 |

---

## 🗺️ 模块导航

| 模块 | 内容概要 | 关键知识点 |
|------|---------|-----------|
|  **Models & Embeddings** | LLMs / ChatModels / 嵌入模型调用规范 | 流式迭代输出、角色标签消息、向量化接口差异 |
| 📝 **Prompt Engineering** | 通用模板 / Few-Shot / ChatPromptTemplate | `format()` vs `invoke()` 边界、动态历史注入、`partial_variables` 固化 |
| 🔗 **Chain & Runnable** | LCEL 链式调用、`Runnable` 协议、并行分流 | `|` 运算符重载、数据契约对齐、嵌套链设计 |
| 🔄 **Output Parsers** | `StrOutputParser` / `JsonOutputParser` / `RunnableLambda` | 类型转换、非结构化输出结构化、自定义函数包装 |
| 💾 **Memory** | 短期内存记忆 / 文件级长期记忆实现 | `RunnableWithMessageHistory`、`BaseChatMessageHistory` 继承修复、序列化兼容 |
| 📄 **Loaders & Splitters** | CSV/JSON/PDF/Text 加载器 + 递归分割器 | 编码兼容、流式加载、`chunk_size/overlap` 分块策略 |
| 🗄️ **Vector Stores & RAG** | 内存/Chroma 持久化、检索器适配 | `as_retriever()` 转换、`RunnablePassthrough` 并行注入、完整 RAG 闭环 |

---
## 🚀 快速开始
点击这个！！！ [具体文档](notebood/LangChain_1.0.md) 进入学习笔记！！！

这个README由ai生成的 (O.o) 做个仓库是因为有一些代码可以配套进行参考，希望能帮到你！
