# Import Flask and necessary components
from flask import Flask, request, jsonify
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create Flask application
app = Flask(__name__)

# Initialize your LlamaIndex outside of the request handling to save loading time
documents = SimpleDirectoryReader("data").load_data()
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="phi3", request_timeout=3600.0)
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

@app.route('/query', methods=['POST'])
def handle_query():
    # Get data from POST request
    data = request.get_json()
    query = data.get('query')
    
    # Check if the query is provided
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    # Execute the query using your LlamaIndex query engine
    response = query_engine.query(query)
    
    # Return the response as JSON
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
