# Import necessary classes and functions from the llama_index package
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
# Import the dotenv package to manage environment variables
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# Load environment variables from a .env file into the environment
load_dotenv()

# Load documents from the directory named "data" using SimpleDirectoryReader
try:
    documents = SimpleDirectoryReader("Data").load_data()
except Exception as e:
    print(f"Failed to load documents: {e}")


# Set the embedding model to 'nomic-embed-text' using OllamaEmbedding in the Settings
Settings.embed_model  = HuggingFaceEmbedding(model_name="intfloat/e5-large-v2")


# Configure the LLM (large language model) settings with model 'llama3' and a request timeout of 360.0 seconds
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

# Create an index for the loaded documents using VectorStoreIndex
index = VectorStoreIndex.from_documents(documents)

# Convert the index into a query engine capable of handling search queries
query_engine = index.as_query_engine()

# Execute a query on the query engine asking about the author's childhood activities
try:
    response = query_engine.query("What are the working hours for TechNova's customer service team?")
    # Print the response obtained from the query engine
    print(response)
except Exception as e:
    print(f"Error during query execution: {e}")
