import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

XAI_API_KEY = os.environ.get("XAI_API_KEY")

headers = {
    "Authorization": f"Bearer {XAI_API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "grok-2-latest",
    "messages": [
        {
            "role": "system",
            "content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."
        },
        {
            "role": "user",
            "content": "What is the meaning of life, the universe, and everything?"
        }
    ]
}

response = requests.post(
    "https://api.x.ai/v1/chat/completions",
    headers=headers,
    json=data
)

if response.status_code == 200:
    result = response.json()
    print(result["choices"][0]["message"]["content"])
else:
    print(f"Error: {response.status_code}")
    print(response.text)
