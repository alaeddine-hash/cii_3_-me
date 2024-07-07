from flask import Flask, request, jsonify
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # This enables CORS for all domains on all routes


# Load environment variables from a .env file into the environment
load_dotenv()

# Set the embedding model and LLM settings
Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/e5-large-v2")
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

# Initialize document loading and indexing outside of request handling to save loading time
try:
    documents = SimpleDirectoryReader("Data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
except Exception as e:
    index = None
    query_engine = None
    print(f"Failed to initialize document index: {e}")

# Helper function to serialize NodeWithScore objects or any non-serializable objects
def serialize_response(response):
    if isinstance(response, list):
        return [serialize_response(item) for item in response]
    elif hasattr(response, 'to_dict'):
        return response.to_dict()
    elif hasattr(response, '__dict__'):
        return response.__dict__
    return str(response)  # As a fallback, convert to string if it's not a list or doesn't have a dictionary representation



@app.route('/query', methods=['POST'])
def handle_query():
    if not query_engine:
        return jsonify({'error': 'Indexing not initialized'}), 500

    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    try:
        response = query_engine.query(query)
        print('response = ', response)
        return jsonify({'response': str(response)})
    except Exception as e:
        return jsonify({'error': f'Error during query execution: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True, ssl_context=('/etc/ssl/certs/apache-selfsigned.crt', '/etc/ssl/private/apache-selfsigned.key'))
