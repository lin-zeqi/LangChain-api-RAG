from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    file_path="./data/document.pdf",    # 文件所在路径
    mode='page',       # 采用哪一钟读取模式，有page（按页面划分document）和single（单个的document）
    # password='password'     # 读取文件需要的密码是什么
)

for doc in loader.lazy_load():
    print(doc)

