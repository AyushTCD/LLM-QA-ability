'''This code allows you to ask questions to paul graham on his essay 
called "What I worked on". Model can be changed in line 18 and subsequently 
'''
import time 
import os.path 
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader,StorageContext, load_index_from_storage ,Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama

Persist_DIR = "./storage"

if not os.path.exists(Persist_DIR):
    print("I'm starting over")
    documents = SimpleDirectoryReader("data").load_data()

    start_time = time.time()
    # bge-m3 embedding model
    Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

    # ollama/mistral 
    Settings.llm = Ollama(model="mistral", request_timeout=30.0)


    index = VectorStoreIndex.from_documents(documents)

    index.storage_context.persist(persist_dir=Persist_DIR)
    print("Created Storage")

else:
    storage_context = StorageContext.from_defaults(persist_dir=Persist_DIR)

    # bge-m3 embedding model
    Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

    # ollama/mistral 
    Settings.llm = Ollama(model="mistral", request_timeout=30.0)

    print("Okay so Local Model loaded")

    start_time = time.time()
    index = load_index_from_storage(storage_context)    

query_engine = index.as_query_engine()
response = query_engine.query("What does the author talk about the lesson of unlearning?")
print(response)
end_time = time.time()
print(f"Time taken to print = {end_time - start_time} seconds")