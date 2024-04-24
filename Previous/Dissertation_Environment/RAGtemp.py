from llama_index.core import (VectorStoreIndex, SimpleDirectoryReader, StorageContext,
                              load_index_from_storage, Settings)
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama
import os 

PERSIST_DIR = "./storage"
DATA_DIR = "./data" # Stores the Embeddings 

def load_or_create_index():
    if not os.path.exists(PERSIST_DIR):
        if not os.path.exists(DATA_DIR):
            print("Data Directory doesn't exist")
            return None 
        
        documents = SimpleDirectoryReader(DATA_DIR).load_data()

        Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
        Settings.llm = Ollama(model="mistral", request_timeout=30.0)

        # Create and persist index
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print("Index created and persisted.")
    else:
        print("Loading index from storage.")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)

        # Reconfigure embedding model and LLM settings
        Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
        Settings.llm = Ollama(model="mistral", request_timeout=30.0)

        # Load index from storage
        index = load_index_from_storage(storage_context)

    return index.as_query_engine()
