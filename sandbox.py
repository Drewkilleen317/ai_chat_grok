import requests

API_KEY = "AIzaSyCwwGYgU-l2iiWOyokvK6sEsfPkemIobko"  # Or: os.environ["GEMINI_API_KEY"]
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro-preview-03-25:generateContent"

headers = {
    "x-goog-api-key": API_KEY,
    "Content-Type": "application/json"
}

payload = {
    "contents": [
        {
            "role": "user",
            "parts": [{"text": "Say hello from Gemini 2.5 Pro!"}]
        }
    ],
    "generationConfig": {
        "temperature": 0.7,
        "topP": 0.9
    }
}

response = requests.post(ENDPOINT, headers=headers, json=payload)
print("Status:", response.status_code)
try:
    response.raise_for_status()
    print("Response:", response.json())
except Exception as e:
    print("Error:", response.text)