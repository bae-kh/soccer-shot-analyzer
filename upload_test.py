import requests
import json

url = "http://localhost:8000/analyze"
file_path = "backend/backend/uploads/video4.mp4"

with open(file_path, "rb") as f:
    files = {"file": f}
    with requests.post(url, files=files, stream=True) as r:
        for line in r.iter_lines():
            if line:
                print(line.decode("utf-8"))
