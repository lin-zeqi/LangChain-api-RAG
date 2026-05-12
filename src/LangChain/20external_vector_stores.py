import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import CSVLoader
load_dotenv()

# 这个数据库类似Python的SQLite
vector_store = Chroma (
    collection_name="test", # 数据库表名
    embedding_function=DashScopeEmbeddings(
        dashscope_api_key=os.getenv("TONGYI_API_KEY")
    ),
    persist_directory="./data"
)


loader = CSVLoader(
    file_path="./data/info.csv",
    encoding="utf-8",
)
documents = loader.load()

vector_store.add_documents(
    documents = documents,
    ids=["id" + str(i) for i in range(1,len(documents)+1)]
)

# vector_store.delete(
#     ["id1","id2"]
# )

res = vector_store.similarity_search(
    "衣服怎么样",
    5,
    filter={"语义标签": "中性评价"}
)

print(res)
print(vector_store)
