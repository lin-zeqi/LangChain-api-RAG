from langchain_community.document_loaders import CSVLoader

lodaer = CSVLoader(
    file_path="data/user.csv",
    csv_args={
        "delimiter":"|",
        "quotechar":'"',
        "fieldnames":["a","b","c"]
    },
    encoding = "utf-8"
)

# documents = lodaer.load()
#
# for document in documents:
#     print(type(document),document)

documents = lodaer.lazy_load()
for document in documents:
    print(type(document),document)