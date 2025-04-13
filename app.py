from time import time
import requests
from pymongo import MongoClient
import streamlit as st

st.set_page_config(
    page_icon="💬", 
    layout="wide", 
    page_title="Grok Chat",
    initial_sidebar_state="expanded",
    menu_items=None)

ss = st.session_state

def get_database():
    # Read MongoDB connection details from Streamlit secrets
    mongodb_url = st.secrets["MONGODB_URL"]
    db_name = st.secrets["MONGODB_DB_NAME"]
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
    ss.show_metrics = True
    ss.llm_avatar = "🤖"
    ss.user_avatar = "😎"
    ss.use_web_search = False
    ss.active_chat = ss.db.chats.find_one({"name": "Scratch Pad"})
    active_model_name = ss.active_chat.get("model")
    model_info = ss.db.models.find_one({"name": active_model_name})
    ss.active_framework = model_info.get("framework")

def get_friendly_time(seconds_ago):
    time_actions = {
        lambda d: d < 60: lambda d: "Just now",
        lambda d: d < 3600: lambda d: f"{int(d / 60)}m ago",
        lambda d: d < 86400: lambda d: f"{int(d / 3600)}h ago",
        lambda d: d < 172800: lambda d: "Yesterday",
        lambda d: d < 604800: lambda d: f"{int(d / 86400)}d ago",
        lambda d: d < 2592000: lambda d: f"{int(d / 604800)}w ago",
        lambda d: True: lambda d: "Long ago"
    }
    for condition, action in time_actions.items():
        if condition(seconds_ago):
            return action(seconds_ago)
            
def update_active_framework():
    """Update the active_framework based on the active chat's model."""
    if 'active_chat' in ss and ss.active_chat:
        active_model_name = ss.active_chat.get("model")
        model_info = ss.db.models.find_one({"name": active_model_name})
        ss.active_framework = model_info.get("framework")

def search_web(query):
    try:
        # Using Serper API for Google Search results
        response = requests.get(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": st.secrets["SERPER_API_KEY"],
                "Content-Type": "application/json"
            },
            params={
                "q": query
            }
        ) 
        if response.status_code != 200:
            st.error(f"Web search failed with status code: {response.status_code}")
            return ""
            
        data = response.json()
        results = []
        
        # Extract organic search results
        organic = data.get('organic', [])
        for result in organic[:3]:
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            link = result.get('link', '')
            if snippet and link:
                results.append(f"[{title}]({link})\n{snippet}")
                
        # Extract knowledge graph if available
        knowledge_graph = data.get('knowledgeGraph', {})
        if knowledge_graph:
            title = knowledge_graph.get('title')
            description = knowledge_graph.get('description')
            link = knowledge_graph.get('link', '')
            if title and description:
                if link:
                    results.insert(0, f"[{title}]({link})\n{description}")
                else:
                    results.insert(0, f"{title}\n{description}")
                
        if not results:
            st.info("No relevant search results found")
            return ""
            
        # Format results as a bulleted list
        context = "\n".join([f"- {result}" for result in results])
        return context        
    except Exception as e:
        st.error(f"Web search error: {str(e)}")
        return ""

def manage_sidebar():
    st.sidebar.markdown("### :blue[Active Chat] 🎯")
    st.sidebar.markdown(f"**Chat Name:** :blue[{ss.active_chat['name']}]")
    st.sidebar.markdown(f"**Model:** :blue[{ss.active_chat['model']}]")
    
    # Web search toggle with state persistence
    web_search = st.sidebar.toggle('Enable Web Search 🔍', value=ss.use_web_search, key='web_search_toggle', help="Enhance responses with web search")
    if web_search != ss.use_web_search:
        ss.use_web_search = web_search
        st.rerun()
    st.sidebar.markdown(f"**Framework:** :blue[{ss.active_framework}]")
    st.sidebar.divider()
    st.sidebar.markdown("### :blue[Select Chat] 📚")
    col1, col2 = st.sidebar.columns([7, 1])
    with col1:
        if st.button("💬 Scratch Pad", key="default_chat", use_container_width=True):
            ss.active_chat = ss.db.chats.find_one({"name": "Scratch Pad"})
            update_active_framework()
            st.rerun()
    with col2:
        if st.button("🧹", key="clear_default", help="Clear Scratch Pad history"):
            ss.db.chats.update_one({"name": "Scratch Pad"},{"$set": {"messages": []}})
            st.rerun()

    # Create and sense if clicked the current chat if not the default chat (Scratch Pad) And the clear button
    if ss.active_chat["name"] != "Scratch Pad":
        friendly_time = get_friendly_time(time() - ss.active_chat.get('created_at'))
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
        friendly_time = get_friendly_time(time() - chat.get('created_at'))
        col1, col2 = st.sidebar.columns([7, 1])
        with col1:
            if st.button(f"💬 {chat['name']} • {friendly_time}", key=f"chat_{chat['name']}", use_container_width=True):
                ss.active_chat = ss.db.chats.find_one({"name": chat["name"]})
                update_active_framework()
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delete_{chat['name']}", help=f"Delete {chat['name']}"):
                ss.db.chats.delete_one({"name": chat["name"]})
                st.rerun()

def get_chat_response():
    # Refresh active_chat to ensure we have the latest messages
    fresh_chat = ss.db.chats.find_one({"name": ss.active_chat['name']})
    messages = fresh_chat["messages"].copy()
    messages.insert(0, {"role": "system", "content": fresh_chat["system_prompt"]})
    
    # Get the last user message
    last_user_message = next((msg for msg in reversed(messages) if msg["role"] == "user"), None)
    
    # Enhance with web search if enabled
    web_results = ""
    if ss.use_web_search and last_user_message:
        web_results = search_web(last_user_message["content"])
        if web_results:
            # Add web search results as a separate message
            search_message = {
                "role": "assistant",
                "content": "🔍 Web Search Results:\n" + web_results,
                "timestamp": time(),
                "is_search_result": True  # Mark this as a search result
            }
            # Add to messages list
            messages.append(search_message)

    # Get the current model information
    model_info = ss.db.models.find_one({"name": fresh_chat["model"]})
    if not model_info:
        st.error(f"Model '{fresh_chat['model']}' not found in database")
        return None
        
    # Get the framework information for this model
    framework_name = model_info.get("framework", "")
    
    if not framework_name:
        st.error(f"Model '{fresh_chat['model']}' has no associated framework")
        return None
    
    start_time = time()
    
    # Import the appropriate framework module
    try:
        # Dynamic import of the framework module
        framework_module = __import__(f"frameworks.{framework_name}", fromlist=["process_chat"])
        
        # Get parameters for the model
        temperature = model_info.get("temperature", 0.7)
        top_p = model_info.get("top_p", 0.9)
        
        # Call the framework-specific processing function
        result = framework_module.process_chat(
            messages=messages,
            model=fresh_chat["model"],
            temperature=temperature,
            top_p=top_p
        )  
    except ImportError:
        st.error(f"Framework module for '{framework_name}' not found")
        return None

    except Exception as e:
        st.error(f"Error processing chat: {str(e)}")
        return None

    # If the framework returned an error
    if "error" in result:
        st.error(f"API Error: {result.get('error')}")
        return None
    
    # Extract the response content
    content = result["content"]
    prompt_tokens = result["prompt_tokens"]
    completion_tokens = result["completion_tokens"]
    elapsed_time = result.get("elapsed_time", time() - start_time)
    
    # Prepare both messages for the chat history
    messages_to_add = []
    
    # If we had search results, add them first
    if web_results:
        messages_to_add.append(search_message)
    
    # Add the AI response
    response_message = {"role": "assistant", "content": content, "timestamp": time()}
    messages_to_add.append(response_message)
    
    # Update MongoDB with all new messages
    ss.db.chats.update_one(
        {"name": ss.active_chat['name']}, 
        {"$push": {"messages": {"$each": messages_to_add}}}
    )
    
    # Calculate cost based on token usage
    tokens = prompt_tokens + completion_tokens
    model_pricing = ss.db.models.find_one({"name": ss.active_chat["model"]}, {"input_price": 1, "output_price": 1, "_id": 0})
    input_cost = (prompt_tokens / 1_000_000) * model_pricing.get("input_price", 0)
    output_cost = (completion_tokens / 1_000_000) * model_pricing.get("output_price", 0)
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
    message_container = st.container(height=565, border=True)
    ss.active_chat = ss.db.chats.find_one({"name": ss.active_chat['name']})
    update_active_framework()
    messages = ss.active_chat.get("messages", [])

    for msg in messages:
        avatar = ss.llm_avatar if msg["role"] == "assistant" else ss.user_avatar
        with message_container.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
    prompt = st.chat_input("Type your message...")
    if prompt:
        message = {"role": "user","content": prompt,"timestamp": time()}
        ss.db.chats.update_one({"name": ss.active_chat['name']}, {"$push": {"messages": message}})
        # Refresh the active_chat after adding the user message
        ss.active_chat = ss.db.chats.find_one({"name": ss.active_chat['name']})
        update_active_framework()
        with message_container.chat_message("user", avatar=ss.user_avatar):
            st.markdown(prompt)
        if response_data := get_chat_response():
            # Display any new messages that were added (search results and AI response)
            fresh_chat = ss.db.chats.find_one({"name": ss.active_chat['name']})
            # Get the last 2 messages if we have search results (identified by is_search_result flag)
            last_messages = fresh_chat["messages"][-2:]
            has_search = any(msg.get("is_search_result") for msg in last_messages)
            new_messages = last_messages if has_search else fresh_chat["messages"][-1:]
            
            for msg in new_messages:
                if msg.get("is_search_result"):
                    with message_container.chat_message("assistant", avatar="🔍"):
                        st.markdown(msg["content"])
                else:
                    with message_container.chat_message("assistant", avatar=ss.llm_avatar):
                        st.markdown(msg["content"])
            # Refresh the active_chat after getting response
            ss.active_chat = ss.db.chats.find_one({"name": ss.active_chat['name']})
            update_active_framework()
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
        
        try:
            db_prompts = list(ss.db.prompts.find())
            available_prompts = [(p["name"], p["content"]) for p in db_prompts]
        except Exception as e:
            st.error(f"Error fetching prompts: {str(e)}")
            available_prompts = []
            
        selected_prompt = st.selectbox(
            "Select System Prompt",
            options=[p[0] for p in available_prompts],
            help="Choose the system prompt that defines how the AI should behave"
        )
        
        # Show the content of the selected prompt
        if selected_prompt:
            prompt_content = next((p[1] for p in available_prompts if p[0] == selected_prompt), "")
            st.text_area("System Prompt Content", value=prompt_content, disabled=True, height=150)
        
        submitted = st.form_submit_button("Create Chat", use_container_width=True)
        if submitted:
            if not new_chat_name:
                st.error("Please enter a chat name")
            elif ss.db.chats.find_one({"name": new_chat_name}):
                st.error("A chat with this name already exists")
            else:
                current_time = time()
                
                # Get the selected prompt content
                system_prompt = next((p[1] for p in available_prompts if p[0] == selected_prompt), "")
                
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
                    update_active_framework()
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
    
    # Add Model functionality
    if model_action == "Add":
        with st.form("add_model_form", clear_on_submit=True):
            # Basic Model Information
            col_name, col_framework = st.columns(2)
            with col_name:
                model_name = st.text_input("Model Name", placeholder="Enter model name...")
            with col_framework:
                # Get frameworks from database
                try:
                    frameworks = list(ss.db.frameworks.find({}, {"name": 1, "display_name": 1, "_id": 0}))
                    framework_options = [fw["display_name"] for fw in frameworks]
                    framework_map = {fw["display_name"]: fw["name"] for fw in frameworks}
                    
                    if not framework_options:
                        st.error("No frameworks available. Please add frameworks first.")
                        framework_display_name = ""
                    else:
                        framework_display_name = st.selectbox("Framework", options=framework_options)
                except Exception as e:
                    st.error(f"Error loading frameworks: {str(e)}")
                    framework_display_name = ""
                    framework_map = {}
            
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
                    # Get framework name from selected display name
                    framework = framework_map.get(framework_display_name, "")
                    
                    if not framework:
                        st.error("Please select a valid framework")
                        return
                    
                    # Prepare model document
                    new_model = {
                        "name": model_name,
                        "framework": framework,  # Add the framework field
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
                
                # Framework selection
                try:
                    frameworks = list(ss.db.frameworks.find({}, {"name": 1, "display_name": 1, "_id": 0}))
                    framework_options = [fw["display_name"] for fw in frameworks]
                    framework_map = {fw["name"]: fw["display_name"] for fw in frameworks}
                    reverse_framework_map = {fw["display_name"]: fw["name"] for fw in frameworks}
                    
                    current_framework = current_model.get("framework", "")
                    current_framework_display = framework_map.get(current_framework, "")
                    
                    if not framework_options:
                        st.error("No frameworks available. Please add frameworks first.")
                        framework_display_name = ""
                    else:
                        framework_display_name = st.selectbox(
                            "Framework", 
                            options=framework_options,
                            index=framework_options.index(current_framework_display) if current_framework_display in framework_options else 0
                        )
                except Exception as e:
                    st.error(f"Error loading frameworks: {str(e)}")
                    framework_display_name = ""
                    reverse_framework_map = {}
                
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
                    # Get framework name from selected display name
                    framework = reverse_framework_map.get(framework_display_name, "")
                    
                    if not framework:
                        st.error("Please select a valid framework")
                        return
                    
                    # Prepare updated model document
                    updated_model = {
                        "name": model_to_edit,
                        "framework": framework,  # Add the framework field
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
    st.markdown("### System Prompt Management 📝")
    
    # Horizontal radio button for prompt actions
    prompt_action = st.radio(
        "Select Prompt Action", 
        ["Add", "Edit", "Delete"], 
        horizontal=True
    )
    
    st.divider()
    
    # Add Prompt functionality
    if prompt_action == "Add":
        with st.form("add_prompt_form", clear_on_submit=True):
            # Basic Prompt Information
            prompt_name = st.text_input("Prompt Name", placeholder="Enter prompt name...")
            prompt_description = st.text_input("Description", placeholder="Brief description of the prompt...")
            prompt_content = st.text_area(
                "Prompt Content", 
                placeholder="Enter system prompt content...",
                height=300
            )
            
            submitted = st.form_submit_button("Add Prompt")
            
            if submitted:
                if not prompt_name:
                    st.error("Prompt Name is required!")
                elif not prompt_content:
                    st.error("Prompt Content is required!")
                else:
                    # Prepare prompt document
                    new_prompt = {
                        "name": prompt_name,
                        "description": prompt_description,
                        "content": prompt_content,
                        "created_at": time()
                    }
                    
                    # Check if prompt already exists
                    existing_prompt = ss.db.prompts.find_one({"name": prompt_name})
                    if existing_prompt:
                        st.error(f"Prompt '{prompt_name}' already exists!")
                    else:
                        # Insert new prompt
                        ss.db.prompts.insert_one(new_prompt)
                        st.success(f"Prompt '{prompt_name}' added successfully!")
    
    # Edit Prompt functionality
    if prompt_action == "Edit":
        # Retrieve all prompts
        available_prompts = list(prompt["name"] for prompt in ss.db.prompts.find())
        
        if not available_prompts:
            st.warning("No prompts available for editing.")
        else:
            with st.form("edit_prompt_form", clear_on_submit=False):
                # Prompt selection
                prompt_to_edit = st.selectbox(
                    "Select Prompt to Edit", 
                    available_prompts
                )
                
                # Retrieve current prompt details
                current_prompt = ss.db.prompts.find_one({"name": prompt_to_edit})
                
                # Prompt description
                prompt_description = st.text_input(
                    "Description", 
                    value=current_prompt.get("description", "")
                )
                
                # Prompt content
                prompt_content = st.text_area(
                    "Prompt Content", 
                    value=current_prompt.get("content", ""),
                    height=300
                )
                
                submitted = st.form_submit_button("Save Prompt")
                
                if submitted:
                    if not prompt_content:
                        st.error("Prompt Content is required!")
                    else:
                        # Prepare updated prompt document
                        updated_prompt = {
                            "name": prompt_to_edit,
                            "description": prompt_description,
                            "content": prompt_content,
                            "updated_at": time()
                        }
                        
                        # Preserve created_at if it exists
                        if "created_at" in current_prompt:
                            updated_prompt["created_at"] = current_prompt["created_at"]
                        
                        # Update the prompt in the database
                        ss.db.prompts.replace_one({"name": prompt_to_edit}, updated_prompt)
                        st.success(f"Prompt '{prompt_to_edit}' updated successfully!")
    
    # Delete Prompt functionality
    if prompt_action == "Delete":
        # Retrieve all prompts except the default
        available_prompts = list(prompt["name"] for prompt in ss.db.prompts.find({"name": {"$ne": "Default System Prompt"}}))
        
        if not available_prompts:
            st.warning("No prompts available for deletion.")
        else:
            with st.form("delete_prompt_form", clear_on_submit=True):
                prompt_to_delete = st.selectbox(
                    "Select Prompt to Delete", 
                    available_prompts,
                    help="Note: 'Default System Prompt' cannot be deleted"
                )
                
                submitted = st.form_submit_button("Delete Prompt")
                
                if submitted:
                    # Double-check the prompt is not the default
                    if prompt_to_delete == "Default System Prompt":
                        st.error("Cannot delete the 'Default System Prompt'.")
                    else:
                        # Confirm deletion
                        confirm = st.checkbox("I understand this action cannot be undone")
                        if confirm:
                            # Perform deletion
                            result = ss.db.prompts.delete_one({"name": prompt_to_delete})
                            
                            if result.deleted_count > 0:
                                st.success(f"Prompt '{prompt_to_delete}' deleted successfully!")
                            else:
                                st.error(f"Could not delete prompt '{prompt_to_delete}'.")

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
    