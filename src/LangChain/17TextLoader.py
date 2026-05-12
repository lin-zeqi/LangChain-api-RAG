from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n","\n",".","?","!","。","？","！"," ",""],
    length_function=len
)

loader = TextLoader(
    file_path="./data/dately.txt",
    encoding="utf-8"
)
res = loader.load()

split_docs = splitter.split_documents(res)

print(len(split_docs))
for doc in split_docs:
    print("="*20)
    print(doc)
    print("="*20)


# print(type(res),res)
# print(len(res))