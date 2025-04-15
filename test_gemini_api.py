#!/usr/bin/env python3

"""
Simple test script for Gemini API.
This script makes a direct call to the Gemini API with hardwired parameters.
"""

import streamlit as st
import requests

# Hardwired parameters
d_endpoint = st.secrets["GEMINI_ENDPOINT"]
d_api_key = st.secrets["GEMINI_API_KEY"]
d_model = "gemini-1.5-pro"
d_generate_endpoint = f"{d_endpoint}/{d_model}:generateContent"
d_messages = [
    {"role": "user", "parts": [{"text": "Hello, how are you?"}]}
]
d_temperature = 0.7
d_top_p = 0.9

print(f"Listing available models at: {d_endpoint}")

# First, list available models
headers = {
    "x-goog-api-key": d_api_key,
    "Content-Type": "application/json"
}

try:
    response = requests.get(d_endpoint, headers=headers)
    print(f"ListModels Response status code: {response.status_code}")
    if response.status_code == 200:
        models_data = response.json()
        print("Available models:")
        for model in models_data.get('models', []):
            print(f"- {model.get('name', 'Unknown')}: {model.get('displayName', 'No display name')}")
    else:
        print(f"Error listing models: {response.text[:1000]}...")
except Exception as e:
    print(f"Request to list models failed: {str(e)}")

# Then, try to generate content
print(f"\nCalling Gemini API at: {d_generate_endpoint}")

# Prepare request payload
payload = {
    "contents": d_messages,
    "generationConfig": {
        "temperature": d_temperature,
        "topP": d_top_p
    }
}

try:
    response = requests.post(d_generate_endpoint, json=payload, headers=headers)
    print(f"GenerateContent Response status code: {response.status_code}")
    
    if response.status_code == 200:
        response_data = response.json()
        print("Response received successfully:")
        print(response_data['candidates'][0]['content']['parts'][0]['text'])
    else:
        print(f"Error details: {response.text[:1000]}...")
except Exception as e:
    print(f"Request failed: {str(e)}")
