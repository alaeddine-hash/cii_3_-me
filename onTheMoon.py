# Import necessary classes and functions from the llama_index package
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate, Prompt
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

# Set the embedding model to 'intfloat/e5-large-v2' using HuggingFaceEmbedding in the Settings
Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/e5-large-v2")

# Configure the LLM (large language model) settings with model 'llama3' and a request timeout of 360.0 seconds
Settings.llm = Ollama(model="llama3", request_timeout=360.0)
#,system_prompt='write the entire response in Japanese \n'

# Create an index for the loaded documents using VectorStoreIndex
index = VectorStoreIndex.from_documents(documents)
# Persist index to disk
index.storage_context.persist("naval_index")

""" # Load index from the storage context
new_index = load_index_from_storage(storage_context) """
# Define the template for the prompt

def addingLanguage(language='Japanese'):
    result = "Context information is provided below.\n" + "-------------------\n" + "{context_str}\n" + "-------------------\n" + "Based on this information, as an APHAIA sales representative, please answer the following question in" + " " + language + " text only: {query_str}\n"
    return result
template = (addingLanguage('english'))

print(template)

template = (
    "Context information is provided below.\n"
    "-------------------\n"
    "{context_str}\n"
    "-------------------\n"
    "Based on this information, as an APHAIA sales representative, please answer the following question in Hindi text only: {query_str}\n"
) 

qa_template = Prompt(template)
query = "Could you provide a quick overview of APAIA leadership team?"


# Convert the index into a query engine capable of handling search queries
query_engine = index.as_query_engine(text_qa_template=qa_template)


# Execute a query on the query engine
try:
    response = query_engine.query(query)
    # Print the response obtained from the query engine
    print(response)
except Exception as e:
    print(f"Error during query execution: {e}")
