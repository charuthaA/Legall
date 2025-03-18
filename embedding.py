import os
import uuid
import chromadb
import pymupdf
from flask import Flask, request, jsonify
from chromadb.config import Settings
import openai  # assuming OpenAI's API is used for embedding generation
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
@app.route('/')
def home():
    return "Flask is running!"

app.config['UPLOAD_FOLDER'] = './uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize ChromaDB Persistent Client
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings()
)

# Create or get collection for embeddings
collection = chroma_client.get_or_create_collection("embeddings")

# Function to generate embeddings using OpenAI (or another provider)
app.route('/')
def home():
    return "Home Page"

@app.route('/api/data')
def get_data():
    return {"message": "API Data"}

# Printing routes one by one
for rule in app.url_map.iter_rules():
    print(rule)

def generate_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="nreimers/MiniLM-L6-H384-uncased"
    )
    return np.array(response['data'][0]['embedding'])

# Function to chunk large text
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Function to process PDF and store embeddings in Chroma DB
def store_embeddings_in_chroma(file_path, is_pdf=True):
    full_text = ""
    
    if is_pdf:
        doc = pymupdf.open(file_path)
        for page in doc:
            full_text += page.get_text()
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            full_text = file.read()

    chunks = chunk_text(full_text)
    for i, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk)
        embedding_id = str(uuid.uuid4())

        collection.add(
            embeddings=[embedding],
            metadatas=[{"fileName": os.path.basename(file_path), "chunkNo": i + 1}],
            ids=[embedding_id],
            documents=[chunk]
        )

# Function to retrieve or create a ChromaDB collection
def get_collection(name="legal_knowledge_hub"):
    return chroma_client.get_or_create_collection(name=name)

# Function to delete a ChromaDB collection
def delete_collection(name="legal_knowledge_hub"):
    chroma_client.delete_collection(name=name)

# Endpoint to upload PDF or TXT file, process it, and store embeddings in Chroma DB
@app.route('/upload/chroma', methods=['POST'])
def upload_file_chroma():
    if 'file' not in request.files:
        return jsonify({"error": "No file given"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in ['pdf', 'txt']:
            return jsonify({"error": "Only PDF or TXT files are allowed"}), 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        store_embeddings_in_chroma(file_path, is_pdf=(file_extension == 'pdf'))

        return jsonify({"message": "File uploaded and embeddings stored successfully"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to search for embeddings based on query
@app.route('/search/chroma', methods=['POST'])
def search_query_chroma():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    query = data.get('query')
    top_k = data.get('top_k', 2)
    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    try:
        query_embedding = generate_embedding(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas']
        )
        matched_metadatas = results['metadatas']
        matched_documents = results['documents']
        data = [{"metadata": meta, "document": doc} for meta, doc in zip(matched_metadatas, matched_documents)]
        return jsonify({"matches": data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to generate embedding for given text
@app.route('/embedding', methods=['POST'])
def generate_text_embedding():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    text = data.get('text')
    if not text:
        return jsonify({"error": "Missing 'text' parameter"}), 400

    try:
        embedding = generate_embedding(text)
        return jsonify({"embedding": embedding.tolist()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
