import os
import time
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama
import nltk
from nltk.translate.meteor_score import single_meteor_score

# nltk.download('wordnet') # Uncomment this line to download necessary datasets the first time you run this

# Set up the directory for persistent storage
Persist_DIR = "./storage"

# Check if the persistent storage directory exists
if not os.path.exists(Persist_DIR):
    print("Persistent storage directory does not exist. Exiting.")
    exit()

# Load the storage and settings
storage_context = StorageContext.from_defaults(persist_dir=Persist_DIR)
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
Settings.llm = Ollama(model="mistral", request_timeout=30.0)
print("Local model loaded from persistent storage")

# Load the index from storage context
index = load_index_from_storage(storage_context)

# Create a query engine from the index
query_engine = index.as_query_engine()

# Make a query about the Battle of Boyne
question = "Tell me about the Battle of Boyne in about 100 words."
start_time = time.time()
response = query_engine.query(question)
end_time = time.time()

print("Generated response:")
print(response)
print(f"Time taken to query = {end_time - start_time} seconds")

# Ground truth reference for evaluation
ground_truth_reference = "The Battle of the Boyne, a crucial event in 1690, marked a significant clash between the forces of the deposed King James II and King William III, alongside Queen Mary II. This battle, integral to James's efforts to reclaim the English and Scottish thrones, was influenced by broader European geopolitical tensions involving major powers like France and the Papal States. Fought near Drogheda, Ireland, William's victory decisively ended James's campaign, contributing to the Protestant ascendancy in Ireland. The conflict, embedded in sectarian and ethnic strife, echoed past Irish Confederate Wars, with underlying issues of sovereignty, religious freedom, and land ownership at stake."

# Calculate METEOR Score
meteor_score = single_meteor_score(ground_truth_reference, response)
print(f"METEOR Score: {meteor_score}")

