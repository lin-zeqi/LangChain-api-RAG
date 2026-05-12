import os
from dotenv import load_dotenv

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import CSVLoader
load_dotenv()

vector_store = InMemoryVectorStore(
    embedding=DashScopeEmbeddings(
        dashscope_api_key=os.getenv("TONGYI_API_KEY")
    )
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

vector_store.delete(
    ["id1","id2"]
)

res = vector_store.similarity_search(
    "衣服怎么样",
    1
)

print(res)
print(vector_store)
