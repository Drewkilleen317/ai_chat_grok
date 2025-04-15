"""
Anthropic Framework Module
Supports Claude 3 (Sonnet, Opus, Haiku) via Anthropic v1/messages API.
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
    Process a chat using the Anthropic Claude API (Claude 3 Sonnet, Opus, Haiku).

    Args:
        messages: List of message dicts with 'role' and 'content' (roles: 'user', 'assistant', 'system')
        model: Model name (e.g., 'claude-3-sonnet-20240229')
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
        api_key = st.secrets["ANTHROPIC_API_KEY"]
        endpoint = st.secrets.get("ANTHROPIC_ENDPOINT", "https://api.anthropic.com/v1/messages")
        url = endpoint

        # Anthropic API expects roles as 'user' and 'assistant' (system prompt is a separate field)
        system_prompt = None
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                # Anthropic v1/messages supports a system prompt field
                if not system_prompt:
                    system_prompt = msg["content"]
                continue
            if msg["role"] in ("user", "assistant"):
                filtered_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        payload = {
            "model": model,
            "messages": filtered_messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": 1024  # Required by Anthropic API
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        elapsed_time = time.time() - start_time
        if response.status_code != 200:
            response.raise_for_status()
        data = response.json()

        content = data["content"][0]["text"] if data.get("content") else ""
        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)

        return {
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_time": elapsed_time
        }
    except Exception as e:
        print(f"Error in Anthropic API call: {str(e)}")
        return None
