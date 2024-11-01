import requests

url = "http://localhost:8000"

try:
    response = requests.get(url)

    if response.status_code == 200:
        print("Chroma is up and running.")
        print("Response:", response.json())
    else:
        print(f"Chroma responded with status code {response.status_code}")
except requests.exceptions.RequestException as e:
    print("Error connecting to Chroma:", e)
