from flask import Flask, request, render_template
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import PorterStemmer
import nltk
import os

# Initialize Flask app
app = Flask(__name__)

# Ensure NLTK data path
nltk_data_path = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Try to load NLTK punkt, if not available, use RegexpTokenizer
try:
    nltk.data.find('tokenizers/punkt')
    tokenizer = word_tokenize
except LookupError:
    print("NLTK punkt not found. Using RegexpTokenizer instead.")
    tokenizer = RegexpTokenizer(r'\w+').tokenize

# Load precomputed data and models
try:
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    tfidf_matrix = joblib.load('tfidf_matrix.pkl')
    products = joblib.load('processed_products.pkl')
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    exit(1)

# Initialize stemmer for preprocessing
stemmer = PorterStemmer()

def preprocess_text(text):
    """
    Tokenize and stem the input text.
    """
    tokens = tokenizer(text.lower())  # Convert to lowercase for uniformity
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    """
    Search products based on a query.
    """
    query = request.args.get('query', '')

    if not query:
        return render_template('index.html', error='Query parameter is required')

    try:
        # Preprocess the query
        processed_query = preprocess_text(query)

        # Transform the query into the same TF-IDF space
        query_vector = tfidf.transform([processed_query])

        # Compute cosine similarity between query and product metadata
        similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

        # Get top-10 results based on similarity scores
        top_indices = similarity_scores.argsort()[-10:][::-1]

        # Extract product IDs from the top results
        top_products = products.iloc[top_indices][['product_id', 'product_name']].to_dict(orient='records')

        # Render the results in HTML
        return render_template('index.html', query=query, results=top_products)
    except Exception as e:
        print(f"Error processing search: {e}")
        return render_template('index.html', error='An error occurred while processing your search.')

if __name__ == '__main__':
    app.run(debug=True)
