import os
import time
import requests
from dotenv import load_dotenv
from pymongo import MongoClient
import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_icon="💬", 
    layout="wide", 
    page_title="Grok Chat",
    initial_sidebar_state="expanded",
    menu_items=None
)
load_dotenv(override=True)

ss = st.session_state

# ---- Database and Initialization Functions ----
def get_database():
    """
    Connect to MongoDB using credentials from environment variables.
    
    Returns:
        MongoDB database instance for chat storage
    """
    mongodb_url = os.getenv('MONGODB_URL')
    db_name = os.getenv('MONGODB_DB_NAME')
    client = MongoClient(
        mongodb_url,
        maxPoolSize=50,           # Maximum number of connections in the pool
        serverSelectionTimeoutMS=5000,  # Timeout for server selection in milliseconds
        connectTimeoutMS=5000,    # Timeout for initial connection in milliseconds
        retryWrites=True          # Enable automatic retrying of failed writes
    )   
    return client[db_name]

def create_chat(new_chat_name, model, system_prompt=None):
    current_time = time.time()
    
    # Use default system prompt if not provided
    if system_prompt is None:
        system_prompt = ss.default_system_prompt
    
    new_chat = {
        "name": new_chat_name,
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
    ss.initialized = True
    ss.db = get_database()
    ss.default_system_prompt = ss.db.prompts.find_one({"name": "Default System Prompt"})
    ss.default_system_prompt = ss.default_system_prompt["content"]
    ss.show_metrics = True
    ss.llm_avatar = "🤖"
    ss.user_avatar = "😎"
    ss.active_chat = ss.db.chats.find_one({"name": "Scratch Pad"})
    ss.active_model_name = ss.active_chat["model"]
    ss.llm_client = {
        'base_url': "https://api.x.ai/v1/chat/completions",
        'api_key': os.environ.get("XAI_API_KEY"),
        'headers': {
            "Content-Type": "application/json"
        },
        'default_model': ss.active_model_name
    }

def get_friendly_time(timestamp):
    now = time.time()
    diff = now - timestamp
    time_actions = {
        lambda d: d < 60: lambda d: "Just now",
        lambda d: d < 3600: lambda d: f"{int(d / 60)}m ago",
        lambda d: d < 86400: lambda d: f"{int(d / 3600)}h ago",
        lambda d: d < 172800: lambda d: "Yesterday",
        lambda d: d < 604800: lambda d: f"{int(d / 86400)}d ago",
        lambda d: d < 2592000: lambda d: f"{int(d / 604800)}w ago",
        lambda d: True: lambda d: time.strftime('%Y-%m-%d', time.localtime(timestamp))
    }
    
    for condition, action in time_actions.items():
        if condition(diff):
            return action(diff)

def manage_sidebar():
    st.sidebar.markdown("### :blue[Active Chat] 🎯")
    st.sidebar.markdown(f"**Chat Name:** :blue[{ss.active_chat['name']}]")
    st.sidebar.markdown(f"**Model:** :blue[{ss.active_model_name}]")
    st.sidebar.divider()
    
    st.sidebar.markdown("### :blue[Select Chat] 📚")
    # Get active chats, excluding archived ones
    active_chats = list(ss.db.chats.find(
        {"archived": {"$ne": True}}, 
        {"name": 1, "created_at": 1}
    ))
    
    # Ensure Scratch Pad is always included and cannot be archived
    scratch_pad = ss.db.chats.find_one({"name": "Scratch Pad"})
    if scratch_pad:
        active_chats.append(scratch_pad)
        # Ensure Scratch Pad is not archived
        scratch_pad["archived"] = False
    
    # chat_names = [chat["name"] for chat in active_chats]
    chats = active_chats
    
    col1, col2 = st.sidebar.columns([7, 1])
    with col1:
        if st.button("💬 Scratch Pad", key="default_chat", use_container_width=True):
            ss.active_chat = ss.db.chats.find_one({"name": "Scratch Pad"})
            ss.active_model_name = ss.active_chat["model"]
            st.rerun()
    with col2:
        if st.button("🧹", key="clear_default", help="Clear Scratch Pad history"):
            ss.db.chats.update_one({"name": "Scratch Pad"},{"$set": {"messages": []}})
            st.rerun()
    
    if ss.active_chat["name"] != "Scratch Pad":
        friendly_time = get_friendly_time(ss.active_chat.get('created_at'))
        col1, col2 = st.sidebar.columns([7, 1])
        with col1:
            st.button(f"👉 {ss.active_chat['name']} • {friendly_time}", key="current_chat", use_container_width=True)
        with col2:
            if st.button("🧹", key="clear_current", help=f"Clear {ss.active_chat['name']} history"):
                ss.db.chats.update_one({"name": ss.active_chat['name']},{"$set": {"messages": []}})
                st.rerun()
    
    other_chats = [c for c in chats if c["name"] not in ["Scratch Pad", ss.active_chat["name"]]]
    
    # Add a divider if there are other chats
    if other_chats:
        st.sidebar.divider()
    
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

def paint_messages(container):
    messages = ss.db.chats.find_one({"name": ss.active_chat['name']}).get("messages", []) 
    for msg in messages:
        avatar = ss.llm_avatar if msg["role"] == "assistant" else ss.user_avatar
        with container.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

def get_costs_from_response(grok_response, model_name):
    # Extract usage data from the Grok response
    usage = grok_response.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    # Retrieve model pricing from the database
    model = ss.db.models.find_one({"name": model_name})
    input_price_per_million = model.get("input_price", 0)
    output_price_per_million = model.get("output_price", 0)

    # Calculate costs
    input_cost = (prompt_tokens / 1_000_000) * input_price_per_million
    output_cost = (completion_tokens / 1_000_000) * output_price_per_million
    total_cost = input_cost + output_cost

    return total_cost

def get_chat_response():
    system_prompt = ss.active_chat.get("system_prompt")
    model_name = ss.active_chat["model"]
    history = ss.active_chat["messages"].copy()
    history.insert(0, {"role": "system", "content": system_prompt})

    start_time = time.time()
    response = requests.post(
        ss.llm_client['base_url'],
        headers={
            **ss.llm_client['headers'],
            "Authorization": f"Bearer {ss.llm_client['api_key']}"
        },
        json={
            "model": model_name,
            "messages": history
        }
    )
    
    if response.status_code != 200:
        st.error(f"API Error {response.status_code}: {response.text}")
        return None
        
    result = response.json()
    content = result["choices"][0]["message"]["content"]

    message = {
        "role": "assistant",
        "content": content,
        "timestamp": time.time()
    }
    ss.db.chats.update_one({"name": ss.active_chat['name']}, {"$push": {"messages": message}})
    
    # Refresh the active chat in session state
    ss.active_chat = ss.db.chats.find_one({"_id": ss.active_chat["_id"]})
    
    # Calculate metrics
    elapsed_time = time.time() - start_time
    tokens = len(content.split())
    
    # Calculate cost
    total_cost = get_costs_from_response(result, ss.active_model_name)
    
    # Calculate messages per dollar
    messages_per_dollar = 100 / total_cost if total_cost > 0 else 0
    
    return {
        "text": content,
        "time": elapsed_time,
        "tokens": tokens,
        "rate": tokens / elapsed_time if elapsed_time > 0 else 0,
        "cost": total_cost,
        "messages_per_dollar": messages_per_dollar
    }

def render_chat_tab():
    message_container = st.container(height=900, border=True)
    paint_messages(message_container)
    
    # Chat input
    prompt = st.chat_input("Type your message...")
    if prompt:
        try:
            user_message = {"role": "user", "content": prompt,
                "timestamp": time.time()
            }
            message = {
                "role": "user",
                "content": prompt,
                "timestamp": time.time()
            }
            ss.db.chats.update_one({"name": ss.active_chat['name']}, {"$push": {"messages": message}})
            ss.active_chat = ss.db.chats.find_one({"name": ss.active_chat['name']})
        except Exception as e:
            st.error(f"Error saving user message: {str(e)}")
        
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
                    f"Speed: {response_data['rate']:.1f} T/s | "
                    f"Cost: ${response_data['cost']:.4f} | "
                    f"Messages/Dollar: {format(response_data['messages_per_dollar'], ',.0f')}"
                )

def render_new_chat_tab():
    st.markdown("### Create New Chat 🆕")
    with st.form("new_chat_form", clear_on_submit=True):
        new_chat_name = st.text_input(
            "Chat Name",
            placeholder="Enter chat name...",
            help="Enter a unique name for your new chat"
        ).strip()
        try:
            db_models = list(model["name"] for model in ss.db.models.find())
            available_models = db_models if db_models else []
        except Exception as e:
            st.error(f"Error fetching models: {str(e)}")
            available_models = []
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

def render_archive_tab():
    st.markdown("### Archive Management 📂")
    st.markdown("Toggle archive status for your chats. Archived chats won't appear in the sidebar.")
    st.divider()
    
    # Retrieve all chats except Scratch Pad
    all_chats = list(ss.db.chats.find({"name": {"$ne": "Scratch Pad"}}, {"name": 1, "archived": 1}))
    
    # Display each chat with its archive status
    for chat in all_chats:
        col1, col2, col3 = st.columns([3, 3, 2])
        archived_status = chat.get('archived', False)
        with col1:
            st.markdown(f"**Chat Name:** :blue[{chat['name']}]")
        with col2:
            st.markdown(f"**Archived:** :blue[{archived_status}]")
        with col3:
            # Use a checkbox to toggle the archived status
            toggle = st.checkbox("Archived", value=archived_status, key=f"toggle_{chat['name']}", help="Check to archive this chat")
            if toggle != archived_status:
                ss.db.chats.update_one({"_id": chat["_id"]}, {"$set": {"archived": toggle}})
                st.rerun()  # Refresh to update the list

def render_models_tab():
    st.warning("⚠️ Model Management is currently under construction. This feature will be available soon!")

def render_prompts_tab():
    st.warning("⚠️ Prompt Management is currently under construction. This feature will be available soon!")

def manage_menu():
    chat_tab, new_chat_tab, archive_tab, models_tab, prompts_tab = st.tabs([
        "💬 Chat", 
        "🆕 New Chat", 
        "🗂️ Archive", 
        "🤖 Models", 
        "📝 Prompts"
    ])
    with chat_tab:
        render_chat_tab()
    with new_chat_tab:
        render_new_chat_tab()
    with archive_tab:
        render_archive_tab()
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