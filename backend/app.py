import os
import threading
import webbrowser
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for API access (like from JS)

# Paths
PROJECT_ROOT = r"C:\Users\aksha\OneDrive\Desktop\Language Translation Project"
model_path = os.path.join(PROJECT_ROOT, "model", "model.pkl")
vectorizer_path = os.path.join(PROJECT_ROOT, "model", "cv.pkl")

# Load Model & Vectorizer
try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("✅ Model and Vectorizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model or vectorizer: {e}")
    model, vectorizer = None, None

# Root Route - Serve HTML Page
@app.route('/')
def home():
    return render_template('index.html')

# Avoid favicon 404
@app.route('/favicon.ico')
def favicon():
    return '', 204

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model or Vectorizer not loaded properly'}), 500

    try:
        data = request.get_json()
        if 'text' not in data or not data['text'].strip():
            return jsonify({'error': 'No text provided'}), 400

        input_text = data['text']
        input_vector = vectorizer.transform([input_text]).toarray()
        prediction = model.predict(input_vector)

        return jsonify({'language': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    threading.Timer(1.25, open_browser).start()
    app.run(debug=True)
