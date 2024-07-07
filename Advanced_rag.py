# Import necessary classes and functions from the llama_index package
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate, Prompt
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import logging
import os

# Configure logging for better debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from a .env file into the environment
load_dotenv()

# Define a function to load documents with exception handling
def load_documents(directory):
    try:
        documents = SimpleDirectoryReader(directory).load_data()
        logger.info(f"Successfully loaded {len(documents)} documents.")
        return documents
    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        return []

# Load documents from the directory named "data"
documents = load_documents("Data")

# Configure the embedding model and LLM with appropriate settings
Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/e5-large-v2")
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

# Create and persist an index for the loaded documents
def create_and_persist_index(documents, index_name="naval_index"):
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(index_name)
    logger.info(f"Index '{index_name}' created and persisted.")
    return index

index = create_and_persist_index(documents)

# Define a function to generate a dynamic prompt template
def addingLanguage(language='Japanese'):
    return (
        "Context information is provided below.\n"
        "-------------------\n"
        "{context_str}\n"
        "-------------------\n"
        "Based on this information, as an APHAIA sales representative, "
        f"please answer the following question in {language} text only: {{query_str}}\n"
    )

# Create a prompt template
template = addingLanguage('English')
qa_template = Prompt(template)

# Convert the index into a query engine capable of handling search queries
query_engine = index.as_query_engine(text_qa_template=qa_template)

# Define a function to execute a query and handle potential errors
def execute_query(query_engine, query):
    try:
        response = query_engine.query(query)
        logger.info(f"Query executed successfully: {query}")
        return response
    except Exception as e:
        logger.error(f"Error during query execution: {e}")
        return None

# Define a sample query
query = "Could you provide a quick overview of APAIA leadership team?"

# Execute the query
response = execute_query(query_engine, query)

# Print the response obtained from the query engine
if response:
    print(response)

# Define additional functions for caching, batching, and more advanced error handling

def cache_results(query, response, cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{hash(query)}.txt")
    with open(cache_file, 'w') as f:
        f.write(str(response))
    logger.info(f"Response cached for query: {query}")

def batch_queries(query_engine, queries):
    responses = []
    for query in queries:
        response = execute_query(query_engine, query)
        if response:
            responses.append(response)
            cache_results(query, response)
    return responses

# Example usage of batch querying
batch_queries_list = [
    "What are the main products of APHAIA?",
    "Can you provide contact details of APHAIA support team?",
    "What is the mission statement of APHAIA?"
]

batch_responses = batch_queries(query_engine, batch_queries_list)

# Print batch responses
for response in batch_responses:
    print(response)
