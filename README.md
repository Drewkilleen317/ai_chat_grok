# Streamlit AI Chat Application Template

This template provides a foundation for building Streamlit web applications featuring AI chat capabilities, data persistence using MongoDB, and a user interface similar to the `ai_chat_grok` app.

## Features

*   **Streamlit Frontend:** Interactive web UI built with Streamlit.
*   **MongoDB Backend:** Uses MongoDB to store application data (e.g., chat conversations, user settings, application-specific data). Connection details managed via secrets.
*   **Chat Interface:**
    *   Sidebar for managing multiple chat sessions (create, select, clear, delete).
    *   Main area for conversation display.
*   **Tabbed Navigation:** Easily organize different application features into tabs.
*   **AI Integration:** Designed for integration with Large Language Models (LLMs). Requires configuration of API endpoints and keys.
*   **Optional Web Search:** Includes functionality to augment AI responses with web search results (e.g., using Serper API).
*   **Secrets Management:** Uses Streamlit's built-in secrets management via `.streamlit/secrets.toml`.

## Project Structure

```
.
├── .streamlit/
│   └── secrets.toml      # API keys and sensitive configuration (DO NOT COMMIT)
├── venv/                 # Python virtual environment (Git-ignored)
├── app.py                # Main Streamlit application script
├── requirements.txt      # Python dependencies
├── .gitignore            # Git ignore configuration
└── README.md             # This file
```

## Setup and Usage

1.  **Clone/Copy:** Obtain the template files.
2.  **Create Virtual Environment:**
    ```bash
    python3 -m venv venv
    ```
3.  **Activate Environment:**
    ```bash
    source venv/bin/activate
    ```
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Configure Secrets:**
    *   Create or edit the `.streamlit/secrets.toml` file.
    *   Add necessary API keys and configuration, ensuring `MONGODB_URL` and `MONGODB_DB_NAME` are present:
        ```toml
        # --- Core App Configuration ---
        MONGODB_URL="mongodb://localhost:27017/" # Or your MongoDB connection string
        MONGODB_DB_NAME="your_app_db_name"      # Database name for this application

        # --- Required External APIs (Example: Chat & Search) ---
        # Replace with your actual LLM provider details
        XAI_BASE_URL = "YOUR_LLM_API_ENDPOINT"
        XAI_API_KEY = "YOUR_LLM_API_KEY"

        # Required if using web search feature
        SERPER_API_KEY = "YOUR_SERPER_API_KEY"

        # --- Add other secrets as needed by your app ---
        # e.g., OTHER_API_KEY = "..."
        ```
    *   **Important:** Ensure `.streamlit` and `*.toml` are listed in your `.gitignore` file to prevent accidental commits.
6.  **Customize `app.py`:**
    *   Modify the `st.set_page_config` call for your new app's title and icon.
    *   Update the `initialize` function with default settings relevant to your new app.
    *   Adapt the `render_*_tab` functions for your application's specific features and UI.
    *   Adjust the `get_chat_response` function if using different LLMs or modifying the chat logic.
    *   Update database interaction logic (schemas, queries in functions like `get_database`, `manage_sidebar`, etc.) as needed for your application's data model.
7.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

## Customization Points in `app.py`

*   **`st.set_page_config`:** App title, icon, layout.
*   **`initialize`:** Default session state values.
*   **`manage_sidebar`:** Logic for chat management or other sidebar elements.
*   **`render_*_tab` functions:** Content and logic for each UI tab.
*   **`get_chat_response` / `search_web`:** Logic for interacting with external APIs (LLMs, search engines).
*   **Database Functions:** Modify functions interacting with MongoDB collections based on your app's data model.

