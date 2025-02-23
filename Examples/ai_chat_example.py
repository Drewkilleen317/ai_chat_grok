import streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv
import time
import os
import google.generativeai as genai
import warnings
warnings.filterwarnings('ignore', message='.*ScriptRunContext.*')

load_dotenv(override=True)

framework_name = "Gemini"

st.set_page_config(
    page_icon="💬", 
    layout="wide", 
    page_title=f"{framework_name} Chat",
    initial_sidebar_state="expanded",
    menu_items=None
)
ss = st.session_state

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

def save_model(model_data):
    """Save or update a model in the database"""
    existing = ss.db.models.find_one({"name": model_data["name"]})
    if existing:
        ss.db.models.update_one(
            {"name": model_data["name"]},
            {"$set": model_data}
        )
    else:
        ss.db.models.insert_one(model_data)

def initialize():
    ss.initialized = True
    ss.db = get_database()
    ss.active_chat = ss.db.chats.find_one({"name": "Scratch Pad"})
    ss.active_model_name = ss.active_chat["model"] 
    ss.default_system_prompt = ss.db.prompts.find_one({"name": "Default System Prompt"})
    ss.default_system_prompt = ss.default_system_prompt["content"]
    ss.show_metrics = True
    ss.llm_avatar = "🤖"
    ss.user_avatar = "😎"
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

def get_available_gemini_models():
    """Get list of models available in the database"""
    try:
        # Get models already in MongoDB
        db_models = list(model["name"] for model in ss.db.models.find())
        
        return db_models if db_models else []

    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        return []

def get_chat_response(prompt):
    """Get a chat response from the Gemini API"""
    try:
        # Determine system prompt
        system_prompt = ss.active_chat.get("system_prompt")
        
        # Prepare the messages for history
        history = []
        
        # Add conversation history
        for msg in ss.active_chat["messages"][:-1]:  # Exclude the last message (current prompt)
            role = "model" if msg["role"] == "assistant" else "user"  # Gemini only uses "model" and "user"
            history.append({
                "role": role,
                "parts": [msg["content"]]
            })
        
        start_time = time.time()
        
        # Create a new Gemini model instance with system instruction
        model = genai.GenerativeModel(
            ss.active_model_name,
            system_instruction=system_prompt
        )
        
        # Create chat session with history
        chat = model.start_chat(history=history)
        response = chat.send_message(prompt)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Approximate token count based on words (Gemini doesn't provide direct token counts)
        tokens = len(response.text.split())
        
        return {
            "text": response.text,
            "time": elapsed_time,
            "tokens": tokens,
            "rate": tokens / elapsed_time if elapsed_time > 0 else 0
        }
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def manage_sidebar():
    # Active chat section
    st.sidebar.markdown("### :blue[Active Chat] 🎯")
    st.sidebar.markdown(f"**Chat Name:** :blue[{ss.active_chat['name']}]")
    st.sidebar.markdown(f"**Model:** :blue[{ss.active_model_name}]")
    
    st.sidebar.divider()
    
    # Chat selection section
    st.sidebar.markdown("### :blue[Select Chat] 📚")
    # Get chat names and times
    chat_names = ss.db.chats.distinct("name")
    chats = list(ss.db.chats.find({}, {"name": 1, "created_at": 1}))
    
    # Default chat first
    col1, col2 = st.sidebar.columns([7, 1])
    with col1:
        if st.button("💬 Scratch Pad", key="default_chat", use_container_width=True):
            ss.active_chat = ss.db.chats.find_one({"name": "Scratch Pad"})
            ss.active_model_name = ss.active_chat["model"]  # Update model name
            st.rerun()
    with col2:
        if st.button("🧹", key="clear_default", help="Clear Scratch Pad history"):
            ss.active_chat["messages"] = [] 
            ss.db.chats.update_one(
                {"name": "Scratch Pad"},
                {"$set": {"messages": ss.active_chat["messages"]}}
            )
            st.rerun()
    
    # Show active chat if not Scratch Pad
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
    
    # Show other chats
    other_chats = [c for c in chats if c["name"] not in ["Scratch Pad", ss.active_chat["name"]]]
    for chat in other_chats:
        friendly_time = get_friendly_time(chat.get('created_at'))
        col1, col2 = st.sidebar.columns([7, 1])
        with col1:
            if st.button(f"💬 {chat['name']} • {friendly_time}", key=f"chat_{chat['name']}", use_container_width=True):
                ss.active_chat = ss.db.chats.find_one({"name": chat["name"]})
                ss.active_model_name = ss.active_chat["model"]  # Update model name
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delete_{chat['name']}", help=f"Delete {chat['name']}"):
                ss.db.chats.delete_one({"name": chat["name"]})
                if chat["name"] == ss.active_chat["name"]:
                    ss.active_chat = ss.db.chats.find_one({"name": "Scratch Pad"})
                    ss.active_model_name = ss.active_chat["model"]  # Update model name
                st.rerun()

def manage_menu():
    """Manage the main menu area with chat and new chat tabs"""
    chat_tab, new_chat_tab, models_tab, prompts_tab = st.tabs([
        "💬 Chat", 
        "🆕 New Chat",
        "🤖 Models",
        "📝 Prompts"
    ])
    message_container = st.container(height=600, border=True)
    paint_messages(message_container, ss.active_chat["messages"])
    
    if prompt := st.chat_input("Enter your message..."):
        ss.active_chat["messages"].append({"role": "user", "content": prompt})
        with message_container.chat_message("user", avatar=ss.user_avatar):
            st.markdown(prompt)
        
        if response_data := get_chat_response(prompt):
            ss.active_chat["messages"].append(
                {"role": "assistant", "content": response_data["text"]}
            )
            with message_container.chat_message("assistant", avatar=ss.llm_avatar):
                st.markdown(response_data["text"])
            
            st.info(
                f"""Time: {response_data['time']:.2f} sec. | """
                f"""Tokens: {response_data['tokens']} | """
                f"""Speed: {response_data['rate']:.1f} T/s"""
            )
        
        ss.db.chats.update_one(
            {"name": ss.active_chat["name"]},
            {"$set": {"messages": ss.active_chat["messages"]}}
        )

    with new_chat_tab:
        st.markdown("### Create New Chat 🆕")
        
        with st.form("new_chat_form", clear_on_submit=True):
            new_chat_name = st.text_input(
                "Chat Name",
                placeholder="Enter chat name...",
                help="Enter a unique name for your new chat",
                key="chat_name_input"
            ).strip()
            
            # Get available Gemini models
            available_models = get_available_gemini_models()
            model = st.selectbox(
                "Select Model",
                options=available_models,
                help="Choose model - different models have different capabilities and performance characteristics"
            )
            
            # Modify system prompt input to be more explicit
            system_prompt = st.text_area(
                label="System Instruction", 
                value=ss.default_system_prompt,
                help="Provide overarching guidelines for the AI's behavior. This will be used as the system instruction for the entire conversation."
            )
            
            submitted = st.form_submit_button("Create Chat", use_container_width=True)
            if submitted:
                # Validate chat name
                if not new_chat_name:
                    st.error("Please enter a chat name")
                elif ss.db.chats.find_one({"name": new_chat_name}):
                    st.error("A chat with this name already exists")
                else:
                    # Create new chat document
                    new_chat = {
                        "name": new_chat_name,
                        "model": model,
                        "system_prompt": system_prompt,
                        "messages": [],
                        "created_at": time.time()
                    }
                    
                    # Insert new chat into database
                    ss.db.chats.insert_one(new_chat)
                    
                    # Set as active chat
                    ss.active_chat = new_chat
                    ss.active_model_name = model
                    
                    st.success(f"Chat '{new_chat_name}' created successfully!")
                    st.rerun()
    
    with models_tab:
        operation = st.radio("Select Operation", ["Add", "Edit", "Delete"], horizontal=True, key="model_operation")
        
        # Get existing models once
        models = list(ss.db.models.find())
        model_names = [m["name"] for m in models]
        
        # Define operation-specific UI functions
        def handle_add():
            return None, {}  # No model selection, empty initial values
            
        def handle_edit():
            model_to_edit = st.selectbox("Select Model to Edit", options=model_names)
            return model_to_edit, ss.db.models.find_one({"name": model_to_edit}) or {}
            
        def handle_delete():
            model_to_delete = st.selectbox("Select Model to Delete", options=model_names)
            if st.button("Delete Selected Model", type="primary"):
                # Check if model is in use by any chats
                chats_using_model = ss.db.chats.find_one({"model": model_to_delete})
                if chats_using_model:
                    st.error(f"Cannot delete model {model_to_delete} as it is being used by existing chats")
                    return None, {}
                
                ss.db.models.delete_one({"name": model_to_delete})
                st.success(f"Model {model_to_delete} deleted successfully!")
                st.rerun()
            return None, {}

        # Action dictionary for operation handlers
        operation_handlers = {
            "Add": handle_add,
            "Edit": handle_edit,
            "Delete": handle_delete
        }
        
        # Execute selected operation handler
        selected_model, initial_values = operation_handlers[operation]()
        
        # Show form for Add/Edit operations
        if operation in ["Add", "Edit"]:
            with st.form("model_form"):
                if operation == "Add":
                    # For Add, use text input for model name
                    name = st.text_input(
                        "Model Name",
                        help="Enter the Gemini model name (e.g., gemini-pro, gemini-pro-vision)"
                    )
                else:
                    # For Edit, use the existing model name
                    name = initial_values.get("name", "")
                    st.text_input("Model Name", value=name, disabled=True)
                
                framework = "gemini"  # Fixed for this app
                
                col1, col2 = st.columns(2)
                with col1:
                    temperature = st.number_input(
                        "Temperature", 0.0, 1.0, 
                        value=initial_values.get("temperature", 0.7), 
                        step=0.1,
                        help="Controls randomness: 0 is focused, 1 is more creative"
                    )
                    top_k = st.number_input(
                        "Top K", 1, 40, 
                        value=initial_values.get("top_k", 40), 
                        step=1,
                        help="Limits vocabulary for each token choice"
                    )
                
                with col2:
                    top_p = st.number_input(
                        "Top P", 0.0, 1.0, 
                        value=initial_values.get("top_p", 0.8), 
                        step=0.1,
                        help="Controls diversity via nucleus sampling"
                    )
                    max_tokens = st.number_input(
                        "Max Output Tokens", 1, 5_000_000, 
                        value=initial_values.get("max_tokens", 8192), 
                        step=1024,
                        help="Maximum length of response. Gemini models can handle massive contexts of millions of tokens!"
                    )
                
                col1, col2 = st.columns(2)
                with col1:
                    text_input = st.checkbox("Text Input", value=initial_values.get("text_input", True))
                    image_input = st.checkbox("Image Input", value=initial_values.get("image_input", False))
                
                with col2:
                    text_output = st.checkbox("Text Output", value=initial_values.get("text_output", True))
                    streaming = st.checkbox("Streaming", value=initial_values.get("streaming", True))
                
                submit_label = "Update Model" if operation == "Edit" else "Add Model"
                if st.form_submit_button(submit_label):
                    if not name:
                        st.error("Please enter a model name")
                        return
                        
                    if operation == "Add" and name in model_names:
                        st.error(f"Model {name} already exists")
                        return
                        
                    model_data = {
                        "name": name,
                        "framework": framework,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "max_tokens": max_tokens,
                        "text_input": text_input,
                        "image_input": image_input,
                        "text_output": text_output,
                        "streaming": streaming,
                        "created_at": time.time() if operation == "Add" else initial_values.get("created_at")
                    }
                    
                    save_model(model_data)
                    st.success(f"Model {name} {'updated' if operation == 'Edit' else 'added'} successfully!")
                    st.rerun()
    
    with prompts_tab:
        st.warning("⚠️ Prompt Management is currently under construction. This feature will be available soon!")
        
        # # Get existing prompts once
        # prompts = list(ss.db.prompts.find())
        # prompt_names = [p["name"] for p in prompts]
        
        # # Define operation-specific UI functions
        # def handle_add():
        #     return None, {}  # No prompt selection, empty initial values
            
        # def handle_edit():
        #     prompt_to_edit = st.selectbox("Select Prompt to Edit", options=prompt_names)
        #     return prompt_to_edit, ss.db.prompts.find_one({"name": prompt_to_edit}) or {}
            
        # def handle_delete():
        #     prompt_to_delete = st.selectbox("Select Prompt to Delete", options=prompt_names)
        #     if st.button("Delete Selected Prompt", type="primary"):
        #         # Check if prompt is in use as the default system prompt
        #         if prompt_to_delete == "Default System Prompt":
        #             st.error("Cannot delete the Default System Prompt")
        #             return None, {}
                    
        #         ss.db.prompts.delete_one({"name": prompt_to_delete})
        #         st.success(f"Prompt {prompt_to_delete} deleted successfully!")
        #         st.rerun()
        #     return None, {}
        
        # # Operation selection
        # operation = st.radio(
        #     "Select Operation",
        #     options=["Add", "Edit", "Delete"],
        #     horizontal=True
        # )
        
        # # Get initial values based on operation
        # operation_handlers = {
        #     "Add": handle_add,
        #     "Edit": handle_edit,
        #     "Delete": handle_delete
        # }
        
        # selected_prompt, initial_values = operation_handlers[operation]()
        
        # # Only show form for Add/Edit operations
        # if operation != "Delete":
        #     with st.form("prompt_form"):
        #         name = st.text_input(
        #             "Prompt Name",
        #             value=initial_values.get("name", ""),
        #             disabled=operation=="Edit"
        #         )
                    
        #         content = st.text_area(
        #             "Prompt Content",
        #             value=initial_values.get("content", ""),
        #             height=200
        #         )
                    
        #         is_default = st.checkbox(
        #             "Set as Default System Prompt",
        #             value=initial_values.get("is_default", False)
        #         )
                    
        #         submit_label = "Update Prompt" if operation == "Edit" else "Add Prompt"
        #         if st.form_submit_button(submit_label, type="primary"):
        #             if not name:
        #                 st.error("Please enter a prompt name")
        #                 return
                            
        #             if not content:
        #                 st.error("Please enter prompt content")
        #                 return
                            
        #             # Check if prompt already exists by querying the database directly
        #             if operation == "Add" and ss.db.prompts.find_one({"name": name}):
        #                 st.error(f"Prompt {name} already exists")
        #                 return
                        
        #             prompt_data = {
        #                 "name": name,
        #                 "content": content,
        #                 "is_default": is_default,
        #                 "created_at": time.time() if operation == "Add" else initial_values.get("created_at")
        #             }
                        
        #             # If this is set as default, unset any existing default
        #             if is_default:
        #                 ss.db.prompts.update_many(
        #                     {"is_default": True},
        #                     {"$set": {"is_default": False}}
        #                 )
                        
        #             save_prompt(prompt_data)
        #             st.success(f"Prompt {name} {'updated' if operation == 'Edit' else 'added'} successfully!")
        #             st.rerun()

def paint_messages(container, messages):
    """Paint messages in the chat container"""
    for msg in messages:
        avatar = ss.llm_avatar if msg["role"] == "assistant" else ss.user_avatar
        with container.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

def get_friendly_time(timestamp):
    """Convert timestamp to friendly format"""
    if not timestamp:
        return "Unknown"
        
    now = time.time()
    diff = now - timestamp
    
    if diff < 60:  # Less than a minute
        return "Just now"
    elif diff < 3600:  # Less than an hour
        minutes = int(diff / 60)
        return f"{minutes}m ago"
    elif diff < 86400:  # Less than a day
        hours = int(diff / 3600)
        return f"{hours}h ago"
    else:
        days = int(diff / 86400)
        return f"{days}d ago"

def main():
    if 'initialized' not in st.session_state:
        initialize()
    manage_sidebar()
    manage_menu()

if __name__ == "__main__":
    main()