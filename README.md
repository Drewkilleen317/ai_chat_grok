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
- **Model Management**: 
    - Add new models with comprehensive configuration
    - Edit existing model parameters
    - Delete unused models
    - Granular control over model capabilities and pricing

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

## Getting an X.AI API Key

To use the Grok models via X.AI's API, you'll need to obtain an API key:

1. Visit the [X.AI developer platform](https://x.ai) and sign up for an account
2. Navigate to the API section in your account dashboard
3. Create a new API key with appropriate permissions
4. Copy the generated API key
5. Add the key to your `.env` file as `XAI_API_KEY=your_key_here`

Note: X.AI may have specific usage tiers and pricing. Check their current documentation for details.

## Setting Up MongoDB

### Option 1: Local MongoDB Installation

1. **Install MongoDB Community Edition**:
   - **macOS** (using Homebrew):
     ```bash
     brew tap mongodb/brew
     brew install mongodb-community
     ```
   - **Windows**: Download and install from the [MongoDB website](https://www.mongodb.com/try/download/community)
   - **Linux**: Follow distribution-specific instructions from the [MongoDB documentation](https://www.mongodb.com/docs/manual/administration/install-on-linux/)

2. **Start MongoDB service**:
   - **macOS**:
     ```bash
     brew services start mongodb-community
     ```
   - **Windows**: MongoDB should run as a service automatically
   - **Linux**:
     ```bash
     sudo systemctl start mongod
     ```

3. **Verify installation**:
   ```bash
   mongo --version
   mongosh
   ```

4. **Update your `.env` file**:
   ```
   MONGODB_URL=mongodb://localhost:27017
   MONGODB_DB_NAME=chat_grok
   ```

### Option 2: MongoDB Atlas (Cloud)

1. Create a free account at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Set up a new cluster (the free tier is sufficient for most users)
3. Create a database user with read/write permissions
4. Configure network access (IP whitelist)
5. Get your connection string from the Atlas dashboard
6. Update your `.env` file with the connection string:
   ```
   MONGODB_URL=mongodb+srv://<username>:<password>@cluster0.mongodb.net/
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

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.
