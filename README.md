# Grok Chat

A streamlined chat interface for interacting with language models through X.AI's API. This application provides a user-friendly web interface built with Streamlit that allows users to engage in conversations with AI models while storing chat history in MongoDB.

![Grok Chat Interface](https://i.imgur.com/placeholder.png)

## Features

- **Multiple Chat Sessions**: Create and manage separate conversations for different topics
- **Model Selection**: Choose from various available AI models
- **Custom System Prompts**: Set specific instructions for how the AI should respond
- **Chat History**: Persistent storage of all conversations in MongoDB
- **Performance Metrics**: View response time, token count, and cost estimates
- **Archive Management**: Archive old chats to keep your workspace organized
- **Centralized LLM Client Configuration**: Efficient management of API connections
- **Dynamic Message Storage**: Real-time database updates for chat messages

## Requirements

- Python 3.8+
- MongoDB instance (local or cloud-based)
- X.AI API key
- Streamlit
- Requests library

## Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/drewkilleen317/ai_chat_grok.git
    cd ai_chat_grok
    ```

2. **Create a virtual environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On macOS/Linux
    # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**
    Create a `.env` file in the root directory with the following variables:
    ```
    XAI_API_KEY=your_x_ai_api_key
    MONGODB_URL=your_mongodb_connection_string
    MONGODB_DB_NAME=chat_grok
    ```

## Running the Application

```bash
streamlit run app.py
```

## Development

- Uses Python 3.8+ with type hints
- Follows PEP 8 style guidelines
- Utilizes Streamlit for frontend
- MongoDB for persistent storage
- Centralized session state management

## Getting an X.AI API Key

To use the Grok models via X.AI's API, you'll need to obtain an API key.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.
