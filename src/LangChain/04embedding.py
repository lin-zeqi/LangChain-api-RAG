import os
from langchain_community.embeddings import DashScopeEmbeddings

embed = DashScopeEmbeddings(
    model='text-embedding-v1',
    dashscope_api_key=os.getenv("TONGYI_API_KEY")
)

print(embed.embed_query("我喜欢你"))
print(embed.embed_query("我爱你"))
print(embed.embed_documents(['我喜欢你','我稀饭拟]','晚上吃什么']))

 