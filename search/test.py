import requests

response = requests.post(
    "https://api.deepseek.com/v1/tools/search",
    headers={
        "Authorization": "Bearer sk-d550b62b591b47b9b37b7c7da4147719",
    },
    json={
        "query": "GPT 6.1",
        "count": 5,
        "need_content": True,
    },
)
response.raise_for_status()

for result in response.json()["results"]:
    print(result["title"])
    print(result["url"])
    #print(result["summary"])
