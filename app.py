from flask import Flask, request, jsonify, render_template
import pickle
import gensim
import numpy as np
import os

app = Flask(__name__)

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load your ML model
with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

# Load your Word2Vec model
word2vec = gensim.models.Word2Vec.load(os.path.join(BASE_DIR, "word2vec.pkl"))

@app.route("/")
def home():
    return render_template("Twitter UI.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("text", "")
    
    if not data:
        return jsonify({"error": "No text provided"}), 400
    
    words = data.split()
    vectors = [word2vec.wv[word] for word in words if word in word2vec.wv]

    if not vectors:
        return jsonify({"error": "No valid words found in Word2Vec model"}), 400

    feature_vector = np.mean(vectors, axis=0).reshape(1, -1)
    prediction = model.predict(feature_vector)
    
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
