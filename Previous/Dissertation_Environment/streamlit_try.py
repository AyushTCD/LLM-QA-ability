import os
import streamlit as st
from llama_index.core import (VectorStoreIndex, SimpleDirectoryReader, StorageContext,
                              load_index_from_storage, Settings)
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama

# Set up page configuration
st.set_page_config(page_title="Conversational Chatbot with Paul Graham's Essays")

# Define the directory for storing the indexed data
PERSIST_DIR = "./storage"

# Function to load or create index
@st.cache_resource(show_spinner=False)
def load_or_create_index():
    if not os.path.exists(PERSIST_DIR):
        st.write("Indexing essays for the first time, please wait...")
        documents = SimpleDirectoryReader("data").load_data()
        print("I have Loaded the data from storage")
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
@st.cache_resource(show_spinner= False)
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
    conversation_context.append(user_query)  # Append the user query to the conversation context
    
    with st.spinner("Searching through the essays..."):
        # Generate a response based on the entire conversation context
        # Ensure you're joining strings in conversation_context, not Response objects
        context_str = " ".join([str(item) for item in conversation_context])
        response = query_engine.query(context_str)
        
        if hasattr(response, 'text'):  # Check if the response has a 'text' attribute
            response_text = response.text  # Use the actual attribute/method that gives you the response text
        else:
            response_text = str(response)  # Fallback to converting response to string if it doesn't have 'text'
        
        st.write(response_text)
        conversation_context.append(response_text)  # Append the response text to the conversation context
        st.write(f"The current context is {context_str}")
