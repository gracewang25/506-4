from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('stopwords')

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups(subset='all')


# nltk.download('stopwords')

app = Flask(__name__)


# TODO: Fetch dataset, initialize vectorizer and LSA here
newsgroups = fetch_20newsgroups(subset='all')
stop_words = list(stopwords.words('english'))

# Vectorize the dataset using TF-IDF
vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(newsgroups.data)

# Apply LSA (SVD)
svd = TruncatedSVD(n_components=100)  # Reduce dimensionality to 100 components
X_lsa = svd.fit_transform(X)


def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # TODO: Implement search engine here
    # return documents, similarities, indices 
    query_vec = vectorizer.transform([query])  
    query_lsa = svd.transform(query_vec)      
    
    
    similarities = cosine_similarity(query_lsa, X_lsa).flatten()
    
    # Get top 5 documents
    top_indices = similarities.argsort()[-5:][::-1]
    top_docs = [newsgroups.data[i] for i in top_indices]
    top_similarities = similarities[top_indices]
    
    return top_docs, top_similarities, top_indices

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    
    
    return jsonify({
        'documents': documents, 
        'similarities': similarities.tolist(),  
        'indices': indices.tolist()  
    })

if __name__ == '__main__':
    app.run(debug=True)
