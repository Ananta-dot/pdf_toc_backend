import json
import torch
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['RESULTS_FOLDER'] = 'results'

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to find the most relevant sentences
def find_relevant_sentences(query, sentences, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    
    relevant_sentences = []
    for score, idx in zip(top_results[0], top_results[1]):
        relevant_sentences.append((sentences[idx], score.item()))
    return relevant_sentences

@app.route('/query', methods=['POST'])
def process_query():
    data = request.get_json()
    sector = data.get('sector')
    filename = data.get('filename')
    
    if not sector or not filename:
        return jsonify({'error': 'Sector and filename are required'}), 400
    
    file_path = os.path.join(app.config['RESULTS_FOLDER'], f"{filename}.json")
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Extract sentences from the JSON data
    sentences = []
    for section, pages in data['Sections'].items():
        sentences.append(section)

    # Find the most relevant sentences
    relevant_sentences = find_relevant_sentences(sector, sentences)

    # Prepare the response
    results = [{'section': sentence, 'pages': data['Sections'][sentence], 'score': score} for sentence, score in relevant_sentences]
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
