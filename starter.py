# Import necessary classes and functions from the llama_index package
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate
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


# Set the embedding model to 'intfloat/e5-large-v2' using OllamaEmbedding in the Settings
Settings.embed_model  = HuggingFaceEmbedding(model_name="intfloat/e5-large-v2")


# Configure the LLM (large language model) settings with model 'llama3' and a request timeout of 360.0 seconds
Settings.llm = Ollama(model="llama3", request_timeout=360.0)


from llama_index.core import PromptTemplate
system_prompt = f"""You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on
the instructions and context provided."""
# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")


# Create an index for the loaded documents using VectorStoreIndex
index = VectorStoreIndex.from_documents(documents)


""" index.storage_context.persist(persist_dir="storage")
from llama_index.core import StorageContext, load_index_from_storage

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="storage")

# load index
index = load_index_from_storage(storage_context) """


template = (
"plz write all the text only in {language_str} without any translation"
"We have provided context information below. \n"
"-------------------\n"
"{context_str}"
"-------------------\n"
"Based on this information, as an APAIA sales representative, please answer the following question: {query_str}\n"
)

qa_template = PromptTemplate(template)
language = 'Japonais'
query = "生成型 AI の開発とそのビジネスへの影響に関する APHAIA のビジョンとは何ですか?また APIA の使命は何ですか?"
super_prompt = qa_template.format(language_str= language, context_str= index,query_str=query )
# Convert the index into a query engine capable of handling search queries
query_engine = index.as_query_engine(kwargs={
    "prompt" : super_prompt
})

# Execute a query on the query engine asking about the author's childhood activities
try:
    response = query_engine.query(super_prompt)
    # Print the response obtained from the query engine
    print(response)
except Exception as e:
    print(f"Error during query execution: {e}")



""" 
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

# load some documents
documents = SimpleDirectoryReader("./data").load_data()

# initialize client, setting path to save data
db = chromadb.PersistentClient(path="./chroma_db")

# create collection
chroma_collection = db.get_or_create_collection("quickstart")

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# create your index
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
) """



""" 
import pinecone
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.vector_stores.pinecone import PineconeVectorStore

# init pinecone
pinecone.init(api_key="<api_key>", environment="<environment>")
pinecone.create_index("quickstart", dimension=1536, metric="euclidean", pod_type="p1")

# construct vector store and customize storage context
storage_context = StorageContext.from_defaults(vector_store=PineconeVectorStore(pinecone.Index("quickstart")))

# Load documents and build index
documents = SimpleDirectoryReader("1234").load_data()
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context) """





