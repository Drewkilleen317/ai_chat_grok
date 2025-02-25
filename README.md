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

## Requirements

- Python 3.6+
- MongoDB instance (local or cloud-based)
- X.AI API key

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/drewkilleen317/ai_chat_grok.git
   cd ai_chat_grok
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
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

## Database Setup and Migration

To initialize your database with the necessary collections and default documents, or to migrate an existing database, use the provided database setup script:

```bash
python db_setup.py
```

The script will:
1. Connect to your MongoDB instance using credentials from your `.env` file
2. Create the required collections if they don't exist
3. Add default models and system prompts
4. Optionally import data from another database if specified

## Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Create a new chat**
   - Click on the "New Chat" tab
   - Enter a name for your chat
   - Optionally customize the system prompt
   - Click "Create Chat"

3. **Chat with the AI**
   - Type your message in the input field
   - View the AI's response and performance metrics
   - Continue the conversation as needed

4. **Manage chats**
   - Switch between existing chats using the sidebar
   - Archive or delete chats you no longer need
   - Clear chat history with the cleanup button

## Configuration

### Models

The application uses X.AI's API for generating responses. The available models are configured in the database and can be managed through the Models tab (feature coming soon).

### System Prompts

System prompts define the AI's behavior and can be customized for each chat. A default system prompt is provided, but you can create your own in the Prompts tab (feature coming soon).

## Technologies Used

- **Streamlit**: Web interface framework
- **MongoDB**: Database for storing chats and configurations
- **X.AI API**: AI model provider
- **Python**: Programming language

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)
