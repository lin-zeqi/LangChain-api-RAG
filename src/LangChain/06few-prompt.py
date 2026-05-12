from langchain_core.prompts import PromptTemplate,FewShotPromptTemplate
from langchain_community.llms.tongyi import Tongyi
from dotenv import load_dotenv
import os


load_dotenv()

model = Tongyi(
    model="qwen-max",
    api_key=os.getenv("TONGYI_API_KEY")
)

example_template = PromptTemplate.from_template("单词:{word}，反义词:{antonym}")

example_data =[
    {"word":"大","antonym":"小"},
    {"word":"上","antonym":"下"}
]

few_shot = FewShotPromptTemplate(
    example_prompt=example_template,
    examples = example_data,
    prefix="告知我单词的反义词，我提供以下示例",
    suffix="基于示例，告知我这个单词{input_word}反义词是什么?",
    input_variables=['input_word']
)

res = few_shot.invoke(input={"input_word":"左"})

print(res)

chain = few_shot | model
print(chain.invoke(input={"input_word":"左"}))
