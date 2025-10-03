import requests

url = "http://127.0.0.1:5000/predict"
data = {"tweet": "I hate programming with Python!"}

res = requests.post(url, json=data)
print(res.json())
