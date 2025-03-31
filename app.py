from flask import Flask, request, jsonify, render_template
import pickle
import gensim
import numpy as np

app = Flask(__name__)

# Load your ML model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load your Word2Vec model
word2vec = gensim.models.Word2Vec.load("word2vec.model")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["text"]
    
    # Convert text to Word2Vec features
    words = data.split()
    vectors = [word2vec.wv[word] for word in words if word in word2vec.wv]
    
    if not vectors:
        return jsonify({"error": "No valid words found in Word2Vec model"})

    feature_vector = np.mean(vectors, axis=0).reshape(1, -1)
    
    prediction = model.predict(feature_vector)
    
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
