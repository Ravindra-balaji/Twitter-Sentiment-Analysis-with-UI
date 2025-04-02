import os
from flask import Flask, request, jsonify, render_template
import pickle
import gensim
import numpy as np

app = Flask(__name__, template_folder="templates")

# Get the working directory
BASE_DIR = os.getcwd()

# Load your ML model
model_path = os.path.join(BASE_DIR, "model.pkl")
word2vec_path = os.path.join(BASE_DIR, "word2vec.pkl")

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    word2vec = gensim.models.Word2Vec.load(word2vec_path)

except FileNotFoundError:
    print("ERROR: Model files not found! Make sure 'model.pkl' and 'word2vec.pkl' are uploaded.")

@app.route("/")
def home():
    return render_template("Twitter UI.html")  # Ensure this file is inside the 'templates/' folder

@app.route("/predict", methods=["POST"])
def predict():
    try:
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
