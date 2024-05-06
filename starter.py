# Import necessary classes and functions from the llama_index package
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
# Import the dotenv package to manage environment variables
from dotenv import load_dotenv

# Load environment variables from a .env file into the environment
load_dotenv()

# Load documents from the directory named "data" using SimpleDirectoryReader
documents = SimpleDirectoryReader("data").load_data()

# Set the embedding model to 'nomic-embed-text' using OllamaEmbedding in the Settings
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Configure the LLM (large language model) settings with model 'llama3' and a request timeout of 360.0 seconds
Settings.llm = Ollama(model="phi3", request_timeout=3600.0)

# Create an index for the loaded documents using VectorStoreIndex
index = VectorStoreIndex.from_documents(documents)

# Convert the index into a query engine capable of handling search queries
query_engine = index.as_query_engine()

# Execute a query on the query engine asking about the author's childhood activities
response = query_engine.query("if you can resume me the file with  the title Embracing Uncertainty in Startups?")

# Print the response obtained from the query engine
print(response)
