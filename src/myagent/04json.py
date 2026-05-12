import json

a = {
    "name": "zjl",
    "age": 11,
    "gender": "男"
}

s = json.dumps(a, ensure_ascii=False)
print(s)
print(a)

l = [
    {
        "name": "zjl",
        "age": 11,
        "gender": "男"
    },
    {
        "name": "cyl",
        "age": 12,
        "gender": "女"
    },
    {
        "name": "xm",
        "age": 18,
        "gender": "未知"
    }
]
print(json.dumps(l,ensure_ascii=False))


json_str = '{"name": "zjl", "age": 11, "gender": "男"}'

res_dict = json.loads(json_str)

print(res_dict)