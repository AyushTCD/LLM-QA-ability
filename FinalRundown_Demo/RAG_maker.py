'''
Creating a RAG for the movies 
# Oppenheimer 
'''
import time 
import os.path 
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader,StorageContext, load_index_from_storage ,Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama

Persist_DIR = "./storageBarbie"

if not os.path.exists(Persist_DIR):
    print("I'm starting over")
    documents = SimpleDirectoryReader("Data/Barbie").load_data()

    start_time = time.time()
    # bge-m3 embedding model
    Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

    # ollama/mistral 
    Settings.llm = Ollama(model="mistral", request_timeout=30.0)


    index = VectorStoreIndex.from_documents(documents)

    index.storage_context.persist(persist_dir=Persist_DIR)
    print("Created Storage for movie")

else:
    storage_context = StorageContext.from_defaults(persist_dir=Persist_DIR)

    # bge-m3 embedding model
    Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

    # ollama/mistral 
    Settings.llm = Ollama(model="llama2", request_timeout=30.0)

    print("Okay so Local Model loaded for the movie")

    start_time = time.time()
    index = load_index_from_storage(storage_context)    

query_engine = index.as_query_engine()
response = query_engine.query("What does Ken think about the real world when he first enters it?")
print(response)
end_time = time.time()
print(f"Time taken to print = {end_time - start_time} seconds")