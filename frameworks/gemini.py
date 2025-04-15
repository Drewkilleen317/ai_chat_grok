# frameworks/gemini.py

"""
Gemini Framework Module
This module handles chat processing using Google's Gemini API.
"""

import streamlit as st
import requests
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
        base_endpoint = st.secrets["GEMINI_ENDPOINT"]
        endpoint = f"{base_endpoint}/{model}:generateContent"
        print(f"Calling Gemini API at: {endpoint}")


        # Format messages for Gemini API
        formatted_messages = []
        for msg in messages:
            if msg["role"] == "system":
                continue  # Gemini does not support system role
            formatted_messages.append({
                "role": msg["role"],
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
        response = requests.post(endpoint, json=payload, headers=headers)
        print(f"Response status code: {response.status_code}")
        try:
            response.raise_for_status()
        except Exception as e:
            error_text = response.text[:500]
            print(f"Error details: {error_text}...")
            if response.status_code == 404 and "model" in error_text.lower():
                print("Model not found. This could be due to an incorrect model name or lack of access. Try 'gemini-pro' as a fallback.")
            raise

        # Extract and return the response text in a standard format
        response_data = response.json()
        text = response_data['candidates'][0]['content']['parts'][0]['text']
        # Gemini API does not provide token usage; set to 0
        return {
            "content": text,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "elapsed_time": 0
        }

    except Exception as e:
        print(f"Error in Gemini API call: {str(e)}")

        return None
