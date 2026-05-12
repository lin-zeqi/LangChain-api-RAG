from langchain_community.document_loaders import JSONLoader

loader = JSONLoader(
    file_path="data/user_json_lines.json",
    jq_schema=".",
    text_content=False,
    json_lines=True
)

res = loader.load()

print(res)