import os
import time
import warnings
import requests
from dotenv import load_dotenv
from pymongo import MongoClient
import streamlit as st

st.set_page_config(
    page_icon="💬", 
    layout="wide", 
    page_title="Grok Chat",
    initial_sidebar_state="expanded",
    menu_items=None
)
load_dotenv(override=True)

framework_name = "Grok"
ss = st.session_state

# ---- Database and Initialization Functions ----
def get_database():
    mongodb_url = os.getenv('MONGODB_URL')
    db_name = os.getenv('MONGODB_DB_NAME')
    client = MongoClient(
        mongodb_url,
        maxPoolSize=50,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        retryWrites=True
    )   
    return client[db_name]

def get_available_grok_models():
    try:
        db_models = list(model["name"] for model in ss.db.models.find())
        return db_models if db_models else []
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        return []

def save_model(model_data):
    existing = ss.db.models.find_one({"name": model_data["name"]})
    if existing:
        ss.db.models.update_one(
            {"name": model_data["name"]},
            {"$set": model_data}
        )
    else:
        ss.db.models.insert_one(model_data)

def create_chat(name, model=None, system_prompt=None):
    """
    Create a new chat with the given parameters. If a chat with the given name already exists,
    return the existing chat.
    
    Args:
        name (str): Name of the chat
        model (str, optional): Model to use for the chat. If None, uses first available model
        system_prompt (str, optional): System prompt for the chat. If None, uses default
        
    Returns:
        dict: The chat document (either newly created or existing)
    """
    # Check if chat already exists
    existing_chat = ss.db.chats.find_one({"name": name})
    if existing_chat:
        return existing_chat
        
    # Get model if not provided
    if model is None:
        available_models = get_available_grok_models()
        if not available_models:
            st.error("No models available. Please add at least one model.")
            return None
        model = available_models[0]
    
    # Get system prompt if not provided
    if system_prompt is None:
        system_prompt = ss.default_system_prompt
        
    # Create new chat
    current_time = time.time()
    new_chat = {
        "name": name,
        "model": model,
        "system_prompt": system_prompt,
        "messages": [],
        "created_at": current_time,
        "updated_at": current_time,
        "archived": False
    }
    ss.db.chats.insert_one(new_chat)
    return new_chat

def initialize():
    if "initialized" not in ss:
        ss.initialized = True
        ss.db = get_database()
        ss.default_system_prompt = ss.db.prompts.find_one({"name": "Default System Prompt"})
        ss.default_system_prompt = ss.default_system_prompt["content"]
        ss.show_metrics = True
        ss.llm_avatar = "🤖"
        ss.user_avatar = "😎"
        
        # Get or create Scratch Pad chat
        scratch_pad = create_chat("Scratch Pad")
        if scratch_pad:
            ss.active_chat = scratch_pad
            ss.active_model_name = scratch_pad["model"]

def save_user_message(prompt):
    """
    Save the user's message to the current active chat in the database.
    
    Args:
        prompt (str): The user's input message
    """
    try:
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": time.time()
        }
        
        # Update the chat document in the database by adding the new message
        ss.db.chats.update_one(
            {"_id": ss.active_chat["_id"]},
            {"$push": {"messages": user_message}}
        )
        
        # Refresh the active chat in session state
        ss.active_chat = ss.db.chats.find_one({"_id": ss.active_chat["_id"]})
    except Exception as e:
        st.error(f"Error saving user message: {str(e)}")

def paint_messages(container, messages):
    for msg in messages:
        avatar = ss.llm_avatar if msg["role"] == "assistant" else ss.user_avatar
        with container.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

def get_friendly_time(timestamp):
    if not timestamp:
        return "Unknown"
    now = time.time()
    diff = now - timestamp
    if diff < 60:
        return "Just now"
    elif diff < 3600:
        minutes = int(diff / 60)
        return f"{minutes}m ago"
    elif diff < 86400:
        hours = int(diff / 3600)
        return f"{hours}h ago"
    else:
        days = int(diff / 86400)
        return f"{days}d ago"

def manage_sidebar():
    st.sidebar.markdown("### :blue[Active Chat] 🎯")
    st.sidebar.markdown(f"**Chat Name:** :blue[{ss.active_chat['name']}]")
    st.sidebar.markdown(f"**Model:** :blue[{ss.active_model_name}]")
    st.sidebar.divider()
    
    st.sidebar.markdown("### :blue[Select Chat] 📚")
    chat_names = ss.db.chats.distinct("name")
    chats = list(ss.db.chats.find({}, {"name": 1, "created_at": 1}))
    
    col1, col2 = st.sidebar.columns([7, 1])
    with col1:
        if st.button("💬 Scratch Pad", key="default_chat", use_container_width=True):
            ss.active_chat = ss.db.chats.find_one({"name": "Scratch Pad"})
            ss.active_model_name = ss.active_chat["model"]
            st.rerun()
    with col2:
        if st.button("🧹", key="clear_default", help="Clear Scratch Pad history"):
            ss.active_chat["messages"] = []
            ss.db.chats.update_one(
                {"name": "Scratch Pad"},
                {"$set": {"messages": ss.active_chat["messages"]}}
            )
            st.rerun()
    
    if ss.active_chat["name"] != "Scratch Pad":
        friendly_time = get_friendly_time(ss.active_chat.get('created_at'))
        col1, col2 = st.sidebar.columns([7, 1])
        with col1:
            st.button(f"👉 {ss.active_chat['name']} • {friendly_time}", key="current_chat", use_container_width=True)
        with col2:
            if st.button("🧹", key="clear_current", help=f"Clear {ss.active_chat['name']} history"):
                ss.active_chat["messages"] = []
                ss.db.chats.update_one(
                    {"name": ss.active_chat["name"]},
                    {"$set": {"messages": ss.active_chat["messages"]}}
                )
                st.rerun()
    
    other_chats = [c for c in chats if c["name"] not in ["Scratch Pad", ss.active_chat["name"]]]
    for chat in other_chats:
        friendly_time = get_friendly_time(chat.get('created_at'))
        col1, col2 = st.sidebar.columns([7, 1])
        with col1:
            if st.button(f"💬 {chat['name']} • {friendly_time}", key=f"chat_{chat['name']}", use_container_width=True):
                ss.active_chat = ss.db.chats.find_one({"name": chat["name"]})
                ss.active_model_name = ss.active_chat["model"]
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delete_{chat['name']}", help=f"Delete {chat['name']}"):
                ss.db.chats.delete_one({"name": chat["name"]})
                if chat["name"] == ss.active_chat["name"]:
                    ss.active_chat = ss.db.chats.find_one({"name": "Scratch Pad"})
                    ss.active_model_name = ss.active_chat["model"]
                st.rerun()

def get_chat_response():
    try:
        # Fetch the latest chat document from the database
        active_chat = ss.db.chats.find_one({"_id": ss.active_chat["_id"]})
        
        # Extract system prompt
        system_prompt = active_chat.get("system_prompt")
        
        # Build message history
        history = []
        if system_prompt:
            history.append({"role": "system", "content": system_prompt})
        
        # Add messages from the database
        history.extend([
            {"role": msg["role"], "content": msg["content"]}
            for msg in active_chat["messages"]
        ])

        start_time = time.time()
        
        # Make API request
        XAI_API_KEY = os.environ.get("XAI_API_KEY")
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": ss.active_model_name,
                "messages": history
            }
        )
        
        if response.status_code != 200:
            st.error(f"API Error {response.status_code}: {response.text}")
            return None
            
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Save AI's response to the database
        ai_message = {
            "role": "assistant",
            "content": content,
            "timestamp": time.time()
        }
        
        ss.db.chats.update_one(
            {"_id": ss.active_chat["_id"]},
            {"$push": {"messages": ai_message}}
        )
        
        # Refresh the active chat in session state
        ss.active_chat = ss.db.chats.find_one({"_id": ss.active_chat["_id"]})
        
        # Calculate metrics
        elapsed_time = time.time() - start_time
        tokens = len(content.split())
        
        return {
            "text": content,
            "time": elapsed_time,
            "tokens": tokens,
            "rate": tokens / elapsed_time if elapsed_time > 0 else 0
        }
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def render_chat_tab():
    message_container = st.container(height=600, border=True)
    paint_messages(message_container, ss.active_chat["messages"])
    
    # Chat input
    prompt = st.chat_input("Type your message...")
    if prompt:
        save_user_message(prompt)
        
        # Immediately display user message in chat container
        with message_container.chat_message("user", avatar=ss.user_avatar):
            st.markdown(prompt)
            
        # Get AI response
        if response_data := get_chat_response():
            with message_container.chat_message("assistant", avatar=ss.llm_avatar):
                st.markdown(response_data["text"])
            if ss.show_metrics:
                st.info(
                    f"Time: {response_data['time']:.2f}s | "
                    f"Tokens: {response_data['tokens']} | "
                    f"Speed: {response_data['rate']:.1f} T/s"
                )

def render_new_chat_tab():
    st.markdown("### Create New Chat 🆕")
    with st.form("new_chat_form", clear_on_submit=True):
        new_chat_name = st.text_input(
            "Chat Name",
            placeholder="Enter chat name...",
            help="Enter a unique name for your new chat"
        ).strip()
        available_models = get_available_grok_models()
        model = st.selectbox(
            "Select Model",
            options=available_models,
            help="Choose model - different models have different capabilities"
        )
        system_prompt = st.text_area(
            label="System Instruction",
            value=ss.default_system_prompt,
            help="Provide overarching guidelines for the AI's behavior"
        )
        submitted = st.form_submit_button("Create Chat", use_container_width=True)
        if submitted:
            if not new_chat_name:
                st.error("Please enter a chat name")
            elif ss.db.chats.find_one({"name": new_chat_name}):
                st.error("A chat with this name already exists")
            else:
                new_chat = create_chat(new_chat_name, model, system_prompt)
                if new_chat:
                    ss.active_chat = new_chat
                    ss.active_model_name = model
                    st.success(f"Chat '{new_chat_name}' created successfully!")
                    st.rerun()

def render_models_tab():
    st.warning("⚠️ Model Management is currently under construction. This feature will be available soon!")

def render_prompts_tab():
    st.warning("⚠️ Prompt Management is currently under construction. This feature will be available soon!")

def manage_menu():
    chat_tab, new_chat_tab, models_tab, prompts_tab = st.tabs(["💬 Chat", "🆕 New Chat", "🤖 Models", "📝 Prompts"])
    with chat_tab:
        render_chat_tab()
    with new_chat_tab:
        render_new_chat_tab()
    with models_tab:
        render_models_tab()
    with prompts_tab:
        render_prompts_tab()

def main():
    if 'initialized' not in st.session_state:
        initialize()
    manage_sidebar()
    manage_menu()

if __name__ == "__main__":
    main()