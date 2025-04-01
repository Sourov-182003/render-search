from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import PorterStemmer
import nltk
import os
import pickle

# Initialize Flask app with CORS support
app = Flask(__name__)

# Enable CORS for all routes and all origins
CORS(app)

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
    with open('product_recommendations.pkl', 'rb') as f:
        product_associations = pickle.load(f)
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

def recommend_items(added_product, product_associations):
    """Generate recommendations based on association rules."""
    recommendations = []
    for antecedents, consequents_list in product_associations.items():
        if added_product in antecedents:
            for consequents, _ in consequents_list:  # Added confidence unpacking
                recommendations.extend(consequents)
    return list(set(recommendations) - {added_product})

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    """
    Search products based on a query.
    Returns JSON response with product results or renders HTML template.
    """
    query = request.args.get('query', '')

    if not query:
        if request.headers.get('Accept') == 'application/json':
            return jsonify({
                'error': 'Query parameter is required',
                'example': '/search?query=oreo+cookies'
            }), 400
        else:
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

        # Extract product details from the top results
        top_products = products.iloc[top_indices][['product_id', 'product_name']].to_dict(orient='records')

        # Return JSON response or render HTML template based on the request
        if request.headers.get('Accept') == 'application/json':
            return jsonify({
                'query': query,
                'count': len(top_products),
                'results': top_products
            })
        else:
            return render_template('index.html', query=query, results=top_products)

    except Exception as e:
        print(f"Error processing search: {e}")
        if request.headers.get('Accept') == 'application/json':
            return jsonify({
                'error': 'An error occurred while processing your search',
                'details': str(e)
            }), 500
        else:
            return render_template('index.html', error='An error occurred while processing your search.')

@app.route('/cart', methods=['GET', 'POST'])
def cart():
    """
    Get recommendations for a product in the cart.
    Handles both GET and POST requests.
    Returns JSON response with recommended products.
    """
    # Handle both GET and POST requests
    if request.method == 'POST':
        product_name = request.form.get('product_name', '')
    else:  # GET
        product_name = request.args.get('product_name', '')

    if not product_name:
        return jsonify({
            'error': 'Product name is required',
            'example_get': '/cart?product_name=Banana',
            'example_post': 'POST /cart with form-data: product_name=Banana'
        }), 400

    try:
        recommendations = recommend_items(product_name, product_associations)

        if not recommendations:
            return jsonify({
                'message': 'No recommendations found for this product',
                'product': product_name
            }), 200

        return jsonify({
            'product': product_name,
            'recommendations': recommendations
        }), 200

    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return jsonify({
            'error': 'An error occurred while generating recommendations',
            'details': str(e)
        }), 500

@app.route('/api/products/by-ids', methods=['GET'])
def get_products_by_ids():
    """
    Fetch products by their IDs.
    """
    try:
        # Extract product IDs from query parameters (comma-separated)
        ids = request.args.get('ids', '').split(',')
        
        # Convert string IDs to integers (if applicable)
        product_ids = [int(id) for id in ids]
        
        # Filter products based on IDs (replace with your actual logic)
        filtered_products = products[products['product_id'].isin(product_ids)]
        
        return jsonify({
            'products': filtered_products.to_dict(orient='records')
        }), 200
        
    except Exception as e:
        print(f"Error fetching products by IDs: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.after_request
def add_cors_headers(response):
    """
    Add CORS headers to all responses.
    """
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Allow-Methods'] = '*'
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
