"""
Ollama Framework Module
Supports local Llama models via Ollama's OpenAI-compatible API.
"""

import streamlit as st
import requests
import time
from typing import List, Dict, Optional

def process_chat(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> Optional[dict]:
    """
    Process a chat using Ollama's local OpenAI-compatible API for Llama models.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: The model name (e.g., 'llama3', 'llama3:8b', etc.)
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter

    Returns:
        Dictionary with standard response format:
        {
            "content": str,  # The generated text response
            "prompt_tokens": int,  # Number of input tokens
            "completion_tokens": int,  # Number of output tokens
            "elapsed_time": float  # Time taken for the request in seconds
        }
    """
    try:
        endpoint = st.secrets.get("OLLAMA_ENDPOINT", "http://localhost:11434/v1")
        api_key = st.secrets.get("OLLAMA_API_KEY", "")
        url = f"{endpoint}/chat/completions"

        ollama_messages = []
        for msg in messages:
            if msg["role"] not in ("system", "user", "assistant"):
                continue
            ollama_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        payload = {
            "model": model,
            "messages": ollama_messages,
            "temperature": temperature,
            "top_p": top_p
        }

        headers = {
            "Content-Type": "application/json"
        }
        if api_key and api_key != "OLLAMA_API_KEY":
            headers["Authorization"] = f"Bearer {api_key}"

        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        elapsed_time = time.time() - start_time
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return {
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_time": elapsed_time
        }
    except Exception as e:
        print(f"Error in Ollama API call: {str(e)}")
        return None
