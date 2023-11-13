from elasticsearch import Elasticsearch, helpers
from flask import Flask, jsonify, request, render_template_string
from embedding import EmbeddingClient
from search import Search
import numpy as np
from db import create_session
from document import Document
import os

session = create_session()


def fetch_document(filename, chunk_index):
    try:
        file = open('docs/' + filename, 'r')
        contents = file.read()
        file.close()
        return contents
    except FileNotFoundError:
        return None

# Define Flask application
app = Flask(__name__)


def cosine_distance(a, b):
    na = np.array(a)
    nb = np.array(b)
    return 1 - na.dot(nb) / (np.linalg.norm(na) * np.linalg.norm(nb))


@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    limit = request.args.get('limit', 10)
    embedded_query = EmbeddingClient.instance().create(query)
    results = Search().execute(embedded_query, limit=limit)
    return jsonify([{
        'filename': result.filename,
        'chunk_index': result.chunk_index,
        'distance': cosine_distance(embedded_query, result.embedding)
    } for result in results])


@app.route('/document', methods=['GET'])
def retrieve_document():
    filename = request.args.get('filename')
    chunk_index = request.args.get('chunk_index') or 0
    document = fetch_document(filename, chunk_index)
    if document is None:
        return jsonify({
            'error': 'Document not found'
        }), 404
    return jsonify({
        'content': document
    })
