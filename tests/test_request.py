import requests

url = "http://127.0.0.1:8000/predict"
files = {"file": open("test.csv", "rb")}

response = requests.post(url, files=files)
print(response.json())