from flask import Flask, request, jsonify, send_from_directory
import torch
import torch.nn as nn
import os
import torch.nn.functional as F


app = Flask(__name__)

# Serve index.html properly
@app.route("/")
def serve_index():
    return send_from_directory(os.getcwd(), "index.html")

# Initialize a simple MLP model
class MLP(nn.Module):
    def __init__(self, vocab_size=4, embedding_size=10, hidden_size=10):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, vocab_size)
        )

    def forward(self, x):
        return self.mlp(self.wte(x))

# Create the model
model = MLP()

# Letter to index mapping
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
char_to_idx = {c: i for i, c in enumerate(alphabet)}

# Base probabilities (softmax input)
base_logits = torch.zeros(len(alphabet))  # Default is equal probability

@app.route("/update_weights", methods=["POST"])
def update_weights():
    global base_logits

    data = request.json
    connection_count = data.get("connectionCount", {})

    # Adjust logits based on connections
    for char, idx in char_to_idx.items():
        base_logits[idx] = connection_count.get(idx, 0)  # Increase if connected

    # Apply softmax to normalize
    probabilities = F.softmax(base_logits, dim=0).tolist()

    # Return the updated weights and probabilities
    weights = base_logits.tolist()

    return jsonify({"weights": weights, "probabilities": probabilities})


if __name__ == "__main__":
    app.run(debug=True)
