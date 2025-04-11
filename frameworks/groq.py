# frameworks/groq.py
"""
Groq API implementation for chat completions.
"""
import streamlit as st
import requests
import time

def process_chat(messages, model, **params):
    """
    Process a chat request using the Groq API
    
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
    api_key = st.secrets.get("GROQ_API_KEY")
    api_base_url = "https://api.groq.com/openai/v1/chat/completions"
    
    # Map model names if needed (in case app-specific model names don't match Groq's naming)
    groq_models = {
        "llama3-8b-8192": "llama3-8b-8192",
        "llama3-70b-8192": "llama3-70b-8192",
        "mixtral-8x7b-32768": "mixtral-8x7b-32768",
        "gemma-7b-it": "gemma-7b-it"
        # Add more mappings as needed
    }
    
    # If the model name doesn't match a Groq model, try to map it or use a default
    if model not in groq_models.values():
        # Check if we have a mapping for this model name
        if model in groq_models:
            groq_model = groq_models[model]
        else:
            # Default to a common Groq model if no mapping exists
            st.warning(f"Model '{model}' not recognized by Groq, using llama3-8b-8192 instead")
            groq_model = "llama3-8b-8192"
    else:
        groq_model = model
        
    # Map to appropriate Groq model
    
    # Prepare request parameters
    temperature = params.get("temperature", 0.7)
    top_p = params.get("top_p", 0.9)
    
    # Make API request
    start_time = time.time()
    try:
        # Clean and validate messages for Groq API
        cleaned_messages = []
        
        # Process each message to remove unsupported fields

        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                error_msg = f"Invalid message format: {msg}"
                # Invalid message format
                return {
                    "error": error_msg,
                    "content": f"Error: {error_msg}",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "elapsed_time": 0
                }
            
            # Create a new clean message with only the fields Groq supports
            clean_msg = {
                "role": msg["role"],
                "content": msg["content"]
            }
            
            # Ensure only valid roles
            if clean_msg['role'] not in ['system', 'user', 'assistant']:
                clean_msg['role'] = 'user'  # Default to user for invalid roles
                
            cleaned_messages.append(clean_msg)
            
        # Prepare the request payload according to Groq API specs
        # See https://console.groq.com/docs/api-reference
        payload = {
            "model": groq_model,  # Use mapped/validated model name
            "messages": cleaned_messages,  # Use the cleaned messages without timestamp
            "temperature": temperature,
            "top_p": top_p,
            # Use max_tokens parameter (required by Groq)
            "max_tokens": params.get("max_tokens", 1024)
        }
        
        # Send the payload to Groq API
        
        # Make the API request with better error handling
        response = requests.post(
            url=api_base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json=payload,
            timeout=60  # Add timeout to prevent hanging
        )
        
        # Check response status
        # Raise exception for HTTP errors
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        error_msg = f"Error: Unable to connect to Groq API. {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                error_msg += f" Details: {error_details}"
            except:
                error_msg += f" Status: {e.response.status_code}, Response: {e.response.text}"
        
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
            "content": f"Error: Invalid response from Groq API. {str(e)}",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "elapsed_time": end_time - start_time
        }