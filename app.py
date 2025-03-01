import os
from time import time
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

def get_database():
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

def initialize():
    ss.initialized = True
    ss.db = get_database()
    ss.default_system_prompt = ss.db.prompts.find_one({"name": "Default System Prompt"}, {"content": 1, "_id": 0})["content"]
    ss.show_metrics = True
    ss.llm_avatar = "🤖"
    ss.user_avatar = "😎"
    ss.active_chat = ss.db.chats.find_one({"name": "Scratch Pad"})
    ss.llm_client = {
        'base_url': "https://api.x.ai/v1/chat/completions",
        'api_key': os.environ.get("XAI_API_KEY"),
        'headers': {
            "Content-Type": "application/json"
        },
        'default_model': ss.active_chat["model"]
    }

def get_friendly_time(timestamp):
    now = time()
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
    st.sidebar.markdown(f"**Model:** :blue[{ss.active_chat['model']}]")
    st.sidebar.divider()
    st.sidebar.markdown("### :blue[Select Chat] 📚")

   

    col1, col2 = st.sidebar.columns([7, 1])

    # Create and sense if clicked the default chat (Scratch Pad) And the clear button for Scratch Pad
    with col1:
        if st.button("💬 Scratch Pad", key="default_chat", use_container_width=True):
            ss.active_chat = ss.db.chats.find_one({"name": "Scratch Pad"})
            st.rerun()
    with col2:
        if st.button("🧹", key="clear_default", help="Clear Scratch Pad history"):
            ss.db.chats.update_one({"name": "Scratch Pad"},{"$set": {"messages": []}})
            st.rerun()

    # Create and sense if clicked the current chat if not the default chat (Scratch Pad) And the clear button
    if ss.active_chat["name"] != "Scratch Pad":
        friendly_time = get_friendly_time(ss.active_chat.get('created_at'))
        col1, col2 = st.sidebar.columns([7, 1])
        with col1:
            st.button(f"👉 {ss.active_chat['name']} • {friendly_time}", key="current_chat", use_container_width=True)
        with col2:
            if st.button("🧹", key="clear_current", help=f"Clear {ss.active_chat['name']} history"):
                ss.db.chats.update_one({"name": ss.active_chat['name']},{"$set": {"messages": []}})
                st.rerun()

    # Get list all the chats in the DB that are not archived, default, or active
    chats = list(ss.db.chats.find({"archived": False}, {"name": 1, "created_at": 1}))
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
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delete_{chat['name']}", help=f"Delete {chat['name']}"):
                ss.db.chats.delete_one({"name": chat["name"]})
                st.rerun()

def get_chat_response():
    ss.active_chat = ss.db.chats.find_one({"name": ss.active_chat['name']})
    system_prompt = ss.active_chat.get("system_prompt")
    history = ss.active_chat["messages"].copy()
    
    # Prepare messages ensuring each has only role and content
    prepared_history = []
    if system_prompt:
        prepared_history.append({"role": "system", "content": system_prompt})
    
    for msg in history:
        # Ensure only role and content are included
        prepared_msg = {
            "role": msg.get("role"),
            "content": msg.get("content")
        }
        prepared_history.append(prepared_msg)

    start_time = time()
    response = requests.post(
        ss.llm_client['base_url'],
        headers={
            **ss.llm_client['headers'],
            "Authorization": f"Bearer {ss.llm_client['api_key']}"
        },
        json={
            "model": ss.active_chat["model"],
            "messages": prepared_history
        }
    )
    end_time = time()  
    elapsed_time = end_time - start_time

    if response.status_code != 200:
        st.error(f"API Error {response.status_code}: {response.text}")
        return None

    result = response.json()
    content = result["choices"][0]["message"]["content"]
    message = {"role": "assistant","content": content,"timestamp": time()}
    ss.db.chats.update_one({"name": ss.active_chat['name']}, {"$push": {"messages": message}})
    
    usage = result.get("usage")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    tokens = prompt_tokens + completion_tokens

    model_pricing = ss.db.models.find_one({"name": ss.active_chat["model"]}, {"input_price": 1, "output_price": 1, "_id": 0})
    input_cost = (prompt_tokens / 1_000_000) * model_pricing.get("input_price")
    output_cost = (completion_tokens / 1_000_000) * model_pricing.get("output_price")
    total_cost = input_cost + output_cost
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
    messages = ss.db.chats.find_one({"name": ss.active_chat['name']}).get("messages", []) 
    for msg in messages:
        avatar = ss.llm_avatar if msg["role"] == "assistant" else ss.user_avatar
        with message_container.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
    
    # Chat input
    prompt = st.chat_input("Type your message...")
    if prompt:
        try:
            message = {"role": "user","content": prompt,"timestamp": time()}
            ss.db.chats.update_one({"name": ss.active_chat['name']}, {"$push": {"messages": message}})
            
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
                current_time = time()
                
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
                new_chat = ss.db.chats.find_one({"name": new_chat_name})
                if new_chat:
                    ss.active_chat = new_chat
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
    st.markdown("### Model Management 🤖")
    
    # Horizontal radio button for model actions
    model_action = st.radio(
        "Select Model Action", 
        ["Add", "Edit", "Delete"], 
        horizontal=True
    )
    
    st.divider()
    
    # Add Model functionality
    if model_action == "Add":
        with st.form("add_model_form", clear_on_submit=True):
            # Basic Model Information
            model_name = st.text_input("Model Name", placeholder="Enter model name...")
            
            # Model Capabilities
            col1, col2 = st.columns(2)
            with col1:
                text_input = st.checkbox("Text Input", value=True)
                image_input = st.checkbox("Image Input")
                text_output = st.checkbox("Text Output", value=True)
                image_output = st.checkbox("Image Output")
            
            with col2:
                tools = st.checkbox("Tools")
                functions = st.checkbox("Functions")
                thinking = st.checkbox("Thinking")
            
            # Model Parameters
            col3, col4 = st.columns(2)
            with col3:
                temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
                top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
            
            with col4:
                max_input_tokens = st.number_input("Max Input Tokens", min_value=0, value=131072)
                max_output_tokens = st.number_input("Max Output Tokens", min_value=0, value=8192)
            
            # Pricing
            col5, col6 = st.columns(2)
            with col5:
                input_price = st.number_input("Input Price (per million tokens)", min_value=0.0, value=2.0, format="%.2f")
            
            with col6:
                output_price = st.number_input("Output Price (per million tokens)", min_value=0.0, value=10.0, format="%.2f")
            
            submitted = st.form_submit_button("Add Model")
            
            if submitted:
                if not model_name:
                    st.error("Model Name is required!")
                else:
                    # Prepare model document
                    new_model = {
                        "name": model_name,
                        "temperature": temperature,
                        "top_p": top_p,
                        "input_price": input_price,
                        "output_price": output_price,
                        "text_input": text_input,
                        "image_input": image_input,
                        "text_output": text_output,
                        "image_output": image_output,
                        "tools": tools,
                        "functions": functions,
                        "thinking": thinking,
                        "max_output_tokens": max_output_tokens,
                        "max_input_tokens": max_input_tokens,
                        "created_at": time()
                    }
                    
                    # Check if model already exists
                    existing_model = ss.db.models.find_one({"name": model_name})
                    if existing_model:
                        st.error(f"Model '{model_name}' already exists!")
                    else:
                        # Insert new model
                        ss.db.models.insert_one(new_model)
                        st.success(f"Model '{model_name}' added successfully!")
    
    # Edit Model functionality
    if model_action == "Edit":
        # Retrieve all models 
        available_models = list(model["name"] for model in ss.db.models.find())
        
        if not available_models:
            st.warning("No models available for editing.")
        else:
            with st.form("edit_model_form", clear_on_submit=True):
                # Model selection
                model_to_edit = st.selectbox(
                    "Select Model to Edit", 
                    available_models
                )
                
                # Retrieve current model details
                current_model = ss.db.models.find_one({"name": model_to_edit})
                
                # Model Capabilities
                col1, col2 = st.columns(2)
                with col1:
                    text_input = st.checkbox("Text Input", value=current_model.get("text_input"))
                    image_input = st.checkbox("Image Input", value=current_model.get("image_input"))
                    text_output = st.checkbox("Text Output", value=current_model.get("text_output"))
                    image_output = st.checkbox("Image Output", value=current_model.get("image_output"))
                
                with col2:
                    tools = st.checkbox("Tools", value=current_model.get("tools"))
                    functions = st.checkbox("Functions", value=current_model.get("functions"))
                    thinking = st.checkbox("Thinking", value=current_model.get("thinking"))
                
                # Model Parameters
                col3, col4 = st.columns(2)
                with col3:
                    temperature = st.slider(
                        "Temperature", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=current_model.get("temperature"), 
                        step=0.05
                    )
                    top_p = st.slider(
                        "Top P", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=current_model.get("top_p"), 
                        step=0.05
                    )
                
                with col4:
                    max_input_tokens = st.number_input(
                        "Max Input Tokens", 
                        min_value=0, 
                        value=current_model.get("max_input_tokens")
                    )
                    max_output_tokens = st.number_input(
                        "Max Output Tokens", 
                        min_value=0, 
                        value=current_model.get("max_output_tokens")
                    )
                
                # Pricing
                col5, col6 = st.columns(2)
                with col5:
                    input_price = st.number_input(
                        "Input Price (per million tokens)", 
                        min_value=0.0, 
                        value=current_model.get("input_price"), 
                        format="%.2f"
                    )
                
                with col6:
                    output_price = st.number_input(
                        "Output Price (per million tokens)", 
                        min_value=0.0, 
                        value=current_model.get("output_price"), 
                        format="%.2f"
                    )
                
                submitted = st.form_submit_button("Save Model")
                
                if submitted:
                    # Prepare updated model document
                    updated_model = {
                        "name": model_to_edit,
                        "temperature": temperature,
                        "top_p": top_p,
                        "input_price": input_price,
                        "output_price": output_price,
                        "text_input": text_input,
                        "image_input": image_input,
                        "text_output": text_output,
                        "image_output": image_output,
                        "tools": tools,
                        "functions": functions,
                        "thinking": thinking,
                        "max_output_tokens": max_output_tokens,
                        "max_input_tokens": max_input_tokens,
                        "updated_at": time()
                    }
                    
                    # Update the model in the database
                    ss.db.models.replace_one({"name": model_to_edit}, updated_model)
                    st.success(f"Model '{model_to_edit}' updated successfully!")
    
    # Delete Model functionality
    if model_action == "Delete":
        # Retrieve all models except the default
        available_models = list(model["name"] for model in ss.db.models.find({"name": {"$ne": "grok-2-latest"}}))
        
        if not available_models:
            st.warning("No models available for deletion.")
        else:
            with st.form("delete_model_form", clear_on_submit=True):
                model_to_delete = st.selectbox(
                    "Select Model to Delete", 
                    available_models,
                    help="Note: 'grok-2-latest' cannot be deleted"
                )
                
                submitted = st.form_submit_button("Delete Model")
                
                if submitted:
                    # Double-check the model is not the default
                    if model_to_delete == "grok-2-latest":
                        st.error("Cannot delete the default model 'grok-2-latest'.")
                    else:
                        # Confirm deletion
                        confirm = st.checkbox("I understand this action cannot be undone")
                        if confirm:
                            # Perform deletion
                            result = ss.db.models.delete_one({"name": model_to_delete})
                            
                            if result.deleted_count > 0:
                                st.success(f"Model '{model_to_delete}' deleted successfully!")
                            else:
                                st.error(f"Could not delete model '{model_to_delete}'.")

    st.divider()

def render_prompts_tab():
    st.warning("⚠️ Prompt Management is currently under construction. This feature will be available soon!")

def render_publish_tab():
    st.markdown("### Publish 📢")
    st.warning("🚧 Publish Functionality is currently under construction. Stay tuned for exciting features!")
    
    st.markdown("#### Upcoming Features:")
    st.markdown("- AI Editor Review")
    st.markdown("  - Spelling and Grammar Corrections")
    st.markdown("  - Text Editing (Add, Remove, Reorder)")
    st.markdown("  - Content Outlining")
    st.markdown("- Audio Podcast Generation")
    st.markdown("  - Two-Party Discussion Conversion")
    
    st.info("We're working on transforming your chats into polished, professional content!")

def manage_menu():
    tab_actions = {
        "💬 Chat": render_chat_tab,
        "🆕 New Chat": render_new_chat_tab,
        "🗂️ Archive": render_archive_tab,
        "🤖 Models": render_models_tab,
        "📝 Prompts": render_prompts_tab,
        "📢 Publish": render_publish_tab
    }
    
    tabs = st.tabs(list(tab_actions.keys()))
    
    for tab, (label, render_func) in zip(tabs, tab_actions.items()):
        with tab:
            render_func()

def main():
    if 'initialized' not in st.session_state:
        initialize()
    manage_sidebar()
    manage_menu()

if __name__ == "__main__":
    main()