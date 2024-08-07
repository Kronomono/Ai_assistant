import requests
import json

url = "http://localhost:8000/search_and_crawl"
payload = {
    "query": "cosmog evolutions",
    "max_results": 3
}
headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)
print(json.dumps(response.json(), indent=2))