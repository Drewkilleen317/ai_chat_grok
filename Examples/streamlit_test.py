import os
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

st.title("Simple Grok Test")

if st.button("Ask Grok"):
    # Ensure we're using the virtualenv's Python
    import sys
    st.write(f"Using Python from: {sys.executable}")
    
    # Debug environment
    XAI_API_KEY = os.environ.get("XAI_API_KEY")
    st.write(f"API Key exists: {'Yes' if XAI_API_KEY else 'No'}")
    st.write(f"API Key length: {len(XAI_API_KEY) if XAI_API_KEY else 0}")
    
    if not XAI_API_KEY:
        st.error("No API key found in environment!")
        st.stop()
    
    client = OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1",
    )

    completion = client.chat.completions.create(
        model="grok-2-latest",
        messages=[
            {
                "role": "system",
                "content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."
            },
            {
                "role": "user",
                "content": "What is the meaning of life, the universe, and everything?"
            },
        ],
    )
    
    st.write(completion.choices[0].message.content)
