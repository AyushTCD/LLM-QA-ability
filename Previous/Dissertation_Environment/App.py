import os
import streamlit as st
from llama_index.core import (VectorStoreIndex, SimpleDirectoryReader, StorageContext,
                              load_index_from_storage, Settings)
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama

# Set up page configuration
st.set_page_config(page_title="Conversational Chatbot with Paul Graham's Essays")

# Define the directory for storing the indexed data and checking the data directory
PERSIST_DIR = "./storage"
DATA_DIR = "./data"  # Ensure this path is correct relative to your script's execution directory

# Function to load or create index
@st.cache_resource(show_spinner=False)
def load_or_create_index():
    if not os.path.exists(PERSIST_DIR):
        st.write("Indexing essays for the first time, please wait...")
        if not os.path.exists(DATA_DIR):
            st.error("Data directory does not exist. Please ensure you have a '/data' directory with text files.")
            return None
        
        documents = SimpleDirectoryReader(DATA_DIR).load_data()

        # Configure embedding model and LLM settings
        Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
        Settings.llm = Ollama(model="mistral", request_timeout=30.0)

        # Create and persist index
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        st.write("Index created and persisted.")
    else:
        st.write("Loading index from storage.")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)

        # Reconfigure embedding model and LLM settings
        Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
        Settings.llm = Ollama(model="mistral", request_timeout=30.0)

        # Load index from storage
        index = load_index_from_storage(storage_context)

    return index.as_query_engine()

# Initialize or load the conversation context
@st.cache_resource(show_spinner=False)
def get_conversation_context():
    return []

# Load or create the index
query_engine = load_or_create_index()

# User interface for query input
st.title("Chat with Paul Graham's Essays 📖")
user_query = st.text_input("Ask a question about Paul Graham's essay 'What I Worked On':")

# Handle query and display response
if user_query:
    conversation_context = get_conversation_context()
    # Append the user query to the conversation context for display
    conversation_context.append(f"User: {user_query}")
    
    with st.spinner("Searching through the essays..."):
        context_str = " ".join([str(item) for item in conversation_context])
        response = query_engine.query(user_query)  # Use just the user_query for querying
        
        if hasattr(response, 'text'):
            response_text = response.text
        else:
            response_text = str(response)
        
        # Append the chatbot response to the conversation context for display
        conversation_context.append(f"Chatbot: {response_text}")

        # Display the conversation history
        st.write("### Conversation History:")
        for message in conversation_context:
            st.write(message)
