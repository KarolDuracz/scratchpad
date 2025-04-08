from flask import Flask, request, jsonify, send_from_directory
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import math

app = Flask(__name__)

# Serve index.html properly
@app.route("/")
def serve_index():
    return send_from_directory(os.getcwd(), "index.html")
    
# Letter to index mapping (including newline as EOT)
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ\n"  # \n as EOT
char_to_idx = {c: i for i, c in enumerate(alphabet)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

# Initialize a simple MLP model with vocab_size=len(alphabet)
class MLP(nn.Module):
    def __init__(self, vocab_size=len(alphabet), embedding_size=16, context_length=3, hidden_size=64):
        super().__init__()
        self.context_length = context_length
        self.wte = nn.Embedding(vocab_size, embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * context_length, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, vocab_size)
        )

    def forward(self, x):
        #for p in self.parameters():
        #    print( " parameters ", p)
        emb = self.wte(x)  # [B, T, C]
        #print(" EMB ", emb)
        flat = emb.view(emb.size(0), -1)  # Flatten
        #print( " FLAT " , flat )
        #print( " self.mlp(flat) " , self.mlp(flat))
        return self.mlp(flat)

# Create the model
model = MLP()
print(model)
for name, param in model.named_parameters():
    print(f"Param name: {name}")

# Global base_logits for interactive influence (flat vector)
base_logits = torch.zeros(len(alphabet))  # Default is all zeros.

############################################
# Update weights now accepts a contextualGraph from the frontend.
@app.route("/update_weights", methods=["POST"])
def update_weights():
    global base_logits
    data = request.json
    print(data)
    if "contextualGraph" in data:
        contextual_graph = data.get("contextualGraph", {})
        # Build a flat count by summing the (possibly weighted) counts for each target across all contexts.
        new_counts = {str(i): 0 for i in range(len(alphabet))}
        print(new_counts)
        print(contextual_graph.items())
        for ctx_key, targets in contextual_graph.items():
            for target, count in targets.items():
                idx = char_to_idx.get(target)
                if idx is not None:
                    new_counts[str(idx)] += count
        for idx in range(len(alphabet)):
            base_logits[idx] = new_counts.get(str(idx), 0)
            
        print("after : ", new_counts)
    else:
        connection_count = data.get("connectionCount", {})
        for idx in range(len(alphabet)):
            base_logits[idx] = connection_count.get(str(idx), 0)
        
        print(" connection_count ", connection_count)
        
    probabilities = F.softmax(base_logits, dim=0).tolist()
    
    print(" probabilities => ", probabilities)
    print( " ARG MAX = ", torch.tensor(probabilities).argmax())
    
    return jsonify({"weights": base_logits.tolist(), "probabilities": probabilities})

############################################
# Parameter update endpoints remain unchanged.
@app.route("/update_parameters", methods=["POST"])
def update_parameters():
    data = request.json
    layer_name = data.get("layer")  # e.g., "mlp.0"
    param_type = data.get("param")  # "weight" or "bias"
    scale = data.get("scale", 1.0)
    target = f"{layer_name}.{param_type}"
    updated = False
    for name, param in model.named_parameters():
        if name == target:
            with torch.no_grad():
                current_mean = param.mean()
                current_std = param.std() + 1e-8
                desired_std = current_std * (scale ** 0.5)
                new_param = (param - current_mean) / current_std * desired_std + current_mean
                param.copy_(new_param)
            updated = True
            break
    if updated:
        return jsonify({"status": "ok"})
    else:
        return jsonify({"error": f"Parameter {target} not found"}), 400

@app.route("/update_parameters_fc2", methods=["POST"])
def update_parameters_fc2():
    data = request.json
    layer_name = data.get("layer")
    param_type = data.get("param")
    scale = data.get("scale", 1.0)
    target = f"{layer_name}.{param_type}"
    updated = False
    for name, param in model.named_parameters():
        if name == target:
            with torch.no_grad():
                current_mean = param.mean()
                current_std = param.std() + 1e-8
                desired_std = current_std * (scale ** 0.5)
                new_param = (param - current_mean) / current_std * desired_std + current_mean
                param.copy_(new_param)
            updated = True
            break
    if updated:
        return jsonify({"status": "ok"})
    else:
        return jsonify({"error": f"Parameter {target} not found"}), 400

############################################
# Endpoint to get parameters for heatmap visualization.
@app.route("/get_parameters", methods=["GET"])
def get_parameters():
    with torch.no_grad():
        fc1_weights = model.mlp[0].weight.cpu().tolist()  # fc1: mlp.0.weight
        fc2_weights = model.mlp[2].weight.cpu().tolist()  # fc2: mlp.2.weight
    return jsonify({"fc1": fc1_weights, "fc2": fc2_weights})

############################################
# Prediction endpoint.
"""
@app.route("/predict", methods=["POST"])
def predict():
    model.eval()
    data = request.json
    print("-"*80)
    print(data)
    input_sequence = data.get("context", [])
    print(" INPUT SEQ = ", input_sequence)
    idxs = [char_to_idx.get(ch, 0) for ch in input_sequence][-model.context_length:]
    while len(idxs) < model.context_length:
        idxs.insert(0, 0)
    input_tensor = torch.tensor([idxs])
    print( " TOKENS PASS THROUGHT TO MODEL FOR PREDICTION : ", input_tensor)
    logits = model(input_tensor)
    print(logits)
    alpha = 0.05  # Use a small scaling factor for base_logits.
    logits = logits + alpha * base_logits
    probabilities = F.softmax(logits, dim=-1).tolist()[0]
    predicted_idx = torch.argmax(logits, dim=-1).item()
    predicted_letter = idx_to_char[predicted_idx]
    return jsonify({
        "input": input_sequence,
        "predicted": predicted_letter,
        "probabilities": probabilities
    })
"""
"""
@app.route("/predict", methods=["POST"])
def predict():
    model.eval() 
    data = request.json
    print(data)
    input_sequence = data.get("context", [])
    # Use the last context_length letters; pad if needed.
    idxs = [char_to_idx.get(ch, 0) for ch in input_sequence][-model.context_length:]
    while len(idxs) < model.context_length:
        idxs.insert(0, 0)
    input_tensor = torch.tensor([idxs])
    logits = model(input_tensor)
    alpha = 0.05  # scaling factor for base_logits
    logits = logits + alpha * base_logits
    probabilities = F.softmax(logits, dim=-1).tolist()[0]
    predicted_idx = torch.argmax(logits, dim=-1).item()
    predicted_letter = idx_to_char[predicted_idx]
    
    print( " predicted letter : ", predicted_letter)
    
    # Optionally, if a target is provided in the request, calculate loss.
    target_char = data.get("target")
    loss_val = None
    if target_char:
        loss_val, _, _ = calculate_prediction_loss_for_context(input_sequence[-model.context_length:], target_char)
    
    print("-"*80)
    print( " LOSSSSSSSSSSSSS " , loss_val, " => ", target_char)
    
    return jsonify({
        "input": input_sequence,
        "predicted": predicted_letter,
        "probabilities": probabilities,
        "loss": loss_val  # may be null if no target provided
    })
"""

@app.route("/predict", methods=["POST"])
def predict():
    model.eval()
    data = request.json
    print(data)
    
    input_sequence = data.get("context", [])
    
    # Ensure context is exactly model.context_length tokens.
    idxs = [char_to_idx.get(ch, 0) for ch in input_sequence][-model.context_length:]
    while len(idxs) < model.context_length:
        idxs.insert(0, 0)
    input_tensor = torch.tensor([idxs])
    
    # Prepare a dictionary to hold activations.
    activations = {}
    
    # Define a hook generator.
    def get_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu().tolist()
        return hook
    
    # Register hooks on layers of interest.
    hook_handles = []
    hook_handles.append(model.wte.register_forward_hook(get_hook("embedding_output")))
    hook_handles.append(model.mlp[0].register_forward_hook(get_hook("fc1_linear_output")))
    hook_handles.append(model.mlp[2].register_forward_hook(get_hook("logits_output")))
    
    # Run the forward pass.
    logits = model(input_tensor)
    alpha = 0.05
    logits = logits + alpha * base_logits
    probabilities = F.softmax(logits, dim=-1).tolist()[0]
    predicted_idx = torch.argmax(logits, dim=-1).item()
    predicted_letter = idx_to_char[predicted_idx]
    
    print( " predicted letter : ", predicted_letter)
    
    # Remove hooks
    for handle in hook_handles:
        handle.remove()
    
    return jsonify({
        "input": input_sequence,
        "predicted": predicted_letter,
        "probabilities": probabilities,
        "activations": activations
    })


############################################
# Dummy positional encoding endpoint.
@app.route("/get_positional_encoding", methods=["GET"])
def get_positional_encoding():
    d_model = 10
    max_len = len(alphabet)
    pe = torch.zeros(max_len, d_model)
    for pos in range(max_len):
        for i in range(d_model):
            angle = pos / (10000 ** (2 * (i//2) / d_model))
            if i % 2 == 0:
                pe[pos, i] = math.sin(angle)
            else:
                pe[pos, i] = math.cos(angle)
    return jsonify(pe.tolist())

############################################
# Loss calculation function (for debugging).
"""
def calculate_prediction_loss(dataset):
    context_length = 3
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for seq in dataset:
            if len(seq) < context_length + 1:
                continue
            for i in range(context_length, len(seq)):
                input_seq = seq[i-context_length:i]
                target = seq[i]
                input_tensor = torch.tensor([input_seq])
                target_tensor = torch.tensor([target])
                logits = model(input_tensor) + base_logits
                loss = loss_fn(logits, target_tensor)
                total_loss += loss.item()
                count += 1
    avg_loss = total_loss / count if count > 0 else float('nan')
    return avg_loss
"""

def calculate_prediction_loss_for_context(context_seq, target_char):
    """
    Calculate cross-entropy loss for a single prediction given a context and a target.
    
    context_seq: list of characters (should be 3 tokens for a 3-token context)
    target_char: a single character string (the target token)
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    
    # Ensure the context is exactly model.context_length tokens.
    idxs = [char_to_idx.get(ch, 0) for ch in context_seq][-model.context_length:]
    while len(idxs) < model.context_length:
        idxs.insert(0, 0)  # pad with the index for 'A' (or zero) if needed
        
    input_tensor = torch.tensor([idxs])  # shape [1, context_length]
    target_idx = char_to_idx.get(target_char, 0)
    target_tensor = torch.tensor([target_idx])  # shape [1]
    
    # Forward pass
    logits = model(input_tensor) + base_logits  # add interactive influence if desired
    # Compute loss
    loss = loss_fn(logits, target_tensor)
    return loss.item(), logits, target_idx


if __name__ == "__main__":
    app.run(debug=True)
