# frameworks/grok.py
"""
Grok API implementation for chat completions.
"""
import streamlit as st
import requests
import time

def process_chat(messages, model, **params):
    """
    Process a chat request using the Grok API
    
    Args:
        messages: List of message objects with 'role' and 'content'
        model: Model name to use
        **params: Additional parameters like temperature, etc.
        
    Returns:
        Dictionary with standard response format:
        {
            "content": str,  # The generated text response
            "prompt_tokens": int,  # Number of input tokens
            "completion_tokens": int,  # Number of output tokens
            "elapsed_time": float  # Time taken for the request in seconds
        }
    """
    # Get API credentials
    api_key = st.secrets.get("XAI_API_KEY")
    api_base_url = "https://api.x.ai/v1/chat/completions"
    
    # Prepare request parameters
    temperature = params.get("temperature", 0.7)
    top_p = params.get("top_p", 0.9)
    
    # Make API request
    start_time = time.time()
    try:
        # Prepare the payload
        payload = {
            "model": model,
            "messages": messages,  # Grok accepts timestamp and other fields
            "temperature": temperature,
            "top_p": top_p
        }
        
        # Make the API request
        response = requests.post(
            url=api_base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json=payload,
            timeout=60  # Add timeout to prevent hanging
        )
        response.raise_for_status()  # Raise exception for HTTP errors
    except requests.exceptions.RequestException as e:
        error_msg = f"Error: Unable to connect to Grok API. {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                error_msg += f" Details: {error_details}"
            except:
                if e.response.text:
                    error_msg += f" Response: {e.response.text}"
                    
        return {
            "error": str(e),
            "content": error_msg,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "elapsed_time": time.time() - start_time
        }
    
    end_time = time.time()
    
    # Process response
    try:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Extract token usage
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        return {
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_time": end_time - start_time
        }
    except (KeyError, ValueError) as e:
        return {
            "error": str(e),
            "content": f"Error: Invalid response from Grok API. {str(e)}",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "elapsed_time": end_time - start_time
        } 