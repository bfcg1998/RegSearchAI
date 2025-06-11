from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example documents
documents = {
    "reg1.pdf": "This regulation explains the handling of hazardous materials.",
    "reg2.pdf": "This section outlines leave policies for military personnel.",
    "reg3.pdf": "Instructions for equipment maintenance and safety procedures."
}

# Precompute embeddings
doc_names = list(documents.keys())
doc_texts = list(documents.values())
doc_embeddings = model.encode(doc_texts)

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        query = request.form["query"]
        query_embedding = model.encode([query])
        scores = cosine_similarity(query_embedding, doc_embeddings)[0]
        results = sorted(
            zip(doc_names, doc_texts, scores),
            key=lambda x: x[2],
            reverse=True
        )
    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
