#!/usr/bin/env python3
"""
Script to add framework documents to the frameworks collection in the ai_chat_mf database.
This script adds entries for Grok and Groq frameworks.
"""

import streamlit as st
from pymongo import MongoClient
import os
import toml
from time import time

def load_secrets():
    """Load MongoDB connection details from Streamlit secrets file."""
    secrets_path = os.path.join(os.path.dirname(__file__), '.streamlit', 'secrets.toml')
    if os.path.exists(secrets_path):
        return toml.load(secrets_path)
    else:
        raise FileNotFoundError(f"Secrets file not found at {secrets_path}")

def get_database():
    """Connect to the MongoDB database using connection details from secrets."""
    secrets = load_secrets()
    mongodb_url = secrets["MONGODB_URL"]
    db_name = secrets["MONGODB_DB_NAME"]
    
    print(f"Connecting to MongoDB at {mongodb_url}, database: {db_name}")
    client = MongoClient(
        mongodb_url,
        maxPoolSize=50,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        retryWrites=True
    )
    return client[db_name]

def add_frameworks():
    """Add framework documents to the frameworks collection."""
    db = get_database()
    
    # Check if frameworks collection exists
    if "frameworks" not in db.list_collection_names():
        print("Creating frameworks collection...")
    
    # Define frameworks
    frameworks = [
        {
            "name": "grok",
            "display_name": "Grok",
            "description": "xAI's advanced language models",
            "base_url": "https://api.x.ai/v1/chat/completions",
            "api_key_name": "XAI_API_KEY",
            "created_at": time()
        },
        {
            "name": "groq",
            "display_name": "Groq",
            "description": "Super fast inference for LLMs",
            "base_url": "https://api.groq.com/openai/v1/chat/completions",
            "api_key_name": "GROQ_API_KEY",
            "created_at": time()
        }
    ]
    
    # Check for existing frameworks
    for framework in frameworks:
        existing = db.frameworks.find_one({"name": framework["name"]})
        if existing:
            print(f"Framework '{framework['name']}' already exists, updating...")
            db.frameworks.replace_one({"name": framework["name"]}, framework)
        else:
            print(f"Adding new framework: {framework['name']}")
            db.frameworks.insert_one(framework)
    
    # Verify frameworks were added
    count = db.frameworks.count_documents({})
    print(f"\nFramework collection now has {count} documents:")
    
    for framework in db.frameworks.find():
        print(f"- {framework['display_name']} ({framework['name']}): {framework['description']}")
    
    print("\nFrameworks added successfully!")

if __name__ == "__main__":
    add_frameworks()
