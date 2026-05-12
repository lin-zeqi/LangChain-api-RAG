# 📘 LangChain 实战笔记与配套代码

> 以「数据流视角」拆解 LCEL 链式架构，从原型跑通到工程化 RAG 落地

本仓库系统整理了 LangChain 核心组件的学习笔记与完整可运行代码示例。内容以 **组件解耦 + 链式组装** 为主线，覆盖模型调用、提示词工程、Chain 工作流、记忆管理、文档加载、向量检索至完整 RAG 链路搭建。所有代码均基于 Python 与阿里云百炼（DashScope）后端验证，适合作为入门指南、调试参考与工程实践模板。

---

## ️ 核心模块

| 模块 | 内容概要 | 关键知识点 |
|------|---------|-----------|
| 🧠 **Models & Embeddings** | LLMs / ChatModels / 嵌入模型调用规范 | 流式输出、角色标签、向量化接口 |
| 📝 **Prompt Engineering** | 通用模板 / Few-Shot / ChatPromptTemplate | `format()` vs `invoke()`、动态历史注入 |
| 🔗 **Chain & Runnable** | LCEL 链式调用、`Runnable` 协议、并行分流 | `|` 运算符重载、数据契约对齐 |
| 🔄 **Output Parsers** | `StrOutputParser` / `JsonOutputParser` / `RunnableLambda` | 类型转换、非结构化输出结构化 |
| 💾 **Memory** | 短期内存记忆 / 文件级长期记忆实现 | `RunnableWithMessageHistory`、序列化修复 |
| 📄 **Loaders & Splitters** | CSV/JSON/PDF/Text 加载器 + 递归分割器 | 编码兼容、流式加载、分块策略 |
| 🗄️ **Vector Stores** | 内存存储 / Chroma 持久化 / 检索器适配 | `as_retriever()`、元数据过滤 |
|  **RAG Pipeline** | 完整检索增强生成链路 | `RunnablePassthrough` 数据截流与并行注入 |

---

## ✨ 项目亮点

- 🔍 **数据流驱动**：强调“上游吐出什么类型，下游能否接住”，提供完整的类型流转对照表
- ️ **工程避坑指南**：收录实战常见报错与修复方案（如 `tuple has no attribute 'type'` 兼容问题、`format` 破坏链式协议等）
- 📦 **开箱即用**：每个模块均配备独立可运行脚本，目录结构清晰，依赖集中管理
-  **阿里云百炼适配**：示例默认对接 `ChatTongyi` 与 `DashScopeEmbeddings`，替换 Key 即可运行
- 📐 **生产思维前置**：在原型代码中预埋日志、重试、环境变量、Token 监控等工程化注释

---

## 🛠️ 快速开始

### 1. 环境准备
```bash
# 克隆仓库
git clone https://github.com/your-username/langchain-notes-code.git
cd langchain-notes-code

# 创建虚拟环境并安装依赖
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
