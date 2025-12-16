import requests

url = "http://127.0.0.1:5000/decide"
data = {
    "counts": {"N":12,"E":8,"S":20,"W":5},
    "last_served": {"N":10,"E":3,"S":0,"W":20}
}

resp = requests.post(url, json=data)
print(resp.json())
