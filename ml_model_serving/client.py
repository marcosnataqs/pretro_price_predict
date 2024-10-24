import requests

input_data = {
    "input": {
        "pbr": 10.5,
        "brent": 60.2,
        "wti": 58.7,
        "production": 2500000,
        "usd": 5.3,
    }
}

response = requests.post("http://127.0.0.1:8000/predict", json=input_data)

# Print status code and response content
print(f"Status Code: {response.status_code}")
print(f"Response Content: {response.text}")

# Try to parse JSON only if the status code is successful
if response.status_code == 200:
    try:
        print("JSON Response:", response.json())
    except requests.exceptions.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
else:
    print(f"Request failed with status code: {response.status_code}")
