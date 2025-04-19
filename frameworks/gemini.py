# frameworks/gemini.py

"""
Gemini Framework Module
This module handles chat processing using Google's Gemini API.
"""

import streamlit as st
import requests
import time
from typing import Dict, List, Optional


def process_chat(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    top_p: float
) -> Optional[str]:
    """
    Process a chat using the Gemini API.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: The model name (e.g., 'gemini-1.5-pro')
        temperature: Temperature parameter for response randomness
        top_p: Top-p parameter for nucleus sampling
    
    Returns:
        The response text from the model, or None if an error occurs
    """
    try:
        # Get API key from environment variable
        api_key = st.secrets["GEMINI_API_KEY"]
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in Streamlit secrets")

        # Prepare the API endpoint
        def get_gemini_api_url(model_name: str) -> str:
            base_url = st.secrets["GEMINI_ENDPOINT"].rstrip("/")
            # Use v1beta for preview/2.5 models, otherwise v1
            if "preview" in model_name or model_name.startswith("gemini-2.5"):
                api_version = "v1beta"
            else:
                api_version = "v1"
            return f"{base_url}/{api_version}/models/{model_name}:generateContent"

        endpoint = get_gemini_api_url(model)


        # Format messages for Gemini API
        formatted_messages = []
        for msg in messages:
            if msg["role"] == "system":
                continue  # Gemini does not support system role
            role = msg["role"]
            if role not in ["user", "assistant"]:
                raise ValueError("Invalid role. Only 'user' and 'assistant' roles are allowed.")
            if role == "assistant":
                role = "model"
            formatted_messages.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

        # Prepare request payload
        payload = {
            "contents": formatted_messages,
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p
            }
        }

        # Make API request
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        }
        start_time = time.time()
        response = requests.post(endpoint, json=payload, headers=headers)
        elapsed_time = time.time() - start_time
        try:
            response.raise_for_status()
        except Exception as e:
            error_text = response.text[:500]
            if response.status_code == 404 and "model" in error_text.lower():
                print("Model not found. This could be due to an incorrect model name or lack of access. Try 'gemini-pro' as a fallback.")
            raise

        # Extract and return the response text in a standard format
        response_data = response.json()
        text = response_data['candidates'][0]['content']['parts'][0]['text']
        # Estimate token counts (1 token ≈ 4 characters for English text)
        def estimate_tokens(s: str) -> int:
            return max(1, int(len(s) / 4))

        prompt_text = " ".join([msg["content"] for msg in messages if msg["role"] != "system"])
        prompt_tokens = estimate_tokens(prompt_text)
        completion_tokens = estimate_tokens(text)

        return {
            "content": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_time": elapsed_time
        }

    except Exception as e:

        return None
