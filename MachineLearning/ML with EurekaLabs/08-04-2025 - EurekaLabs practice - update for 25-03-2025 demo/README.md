10.04.2025 - f***, learning is ok, but such trips are shooting myself in the foot. Next time I have to verify my "brilliant ideas" before I publish something here... on github.
<hr >
12.04-2025 - I need to clarify something so as not to lose my "observation point on the roadmap". <b>This exercise will mainly focus on </b>:<br />
A bit about floating-point numbers and their precision, and wwhat is the impact of decimal precision on deep networks<br />
A bit about the distribution of numbers, about STD, VAR, MEAN, or the fundamental rules of statistics. Normal distribution https://en.wikipedia.org/wiki/Normal_distribution <br />
A bit about why it works, or for example why lenet recognizes numbers. etc <br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/08-04-2025%20-%20EurekaLabs%20practice%20-%20update%20for%2025-03-2025%20demo/lenet%20with%20pixels.png?raw=true)

So in short, when I started this topic I meant to better understand the mathematical principles behind networks. The idea is to build a small demo with graphs of all this on each layer, insert different architectures (MLP <> mini Transformer), give different data and build an initial intuition of how it works. THIS IS ONLY FOR MY EDUCATIONAL PURPOSES. In general, in the long run, this is the basic idea behind what I'm writing about here. SO I JUST WANT TO SOMEHOW UNDERSTAND THE BASICS.

<hr>
<h2>Still TODO</h2>
Today's post is just to give an idea of ​​what I wrote in my post here
https://github.com/KarolDuracz/scratchpad/tree/main/MachineLearning/ML%20with%20EurekaLabs/25-03-2025%20-%20EurekaLabs%20practice
<br /><br />
Still nothing works here properly. But I'll finish it because it helps me learn a bit about how the network works from the inside, even just trying to do something similar to the MLP with context_length = 3. <b>I threw something, and now I just have to explain it. To be sure if it's nonsense or if it still makes sense and is it scalable. It was a mistake, now the topic needs to be closed. I will try to make a working model in the next few weeks to close the topic. </b> 
<h3>1.</h3> While playing around with ChatGPT 4o I also realized a few things, that what I came up with here is a graph approach to networks, i.e. "graph network".
<h3>2.</h3> Even if I manage to finish it and make it predict exactly like Andrej's MLP that generates names learned from this dataset  https://github.com/EurekaLabsAI/mlp/blob/master/data/train.txt . And when I then add a larger tokenizer like "GPT3.5 base" it probably has nothing to do with how the problem is generalized by neural networks and the backpropagation algorithm, which changes parameters in small steps to reduce the loss function. In general, when I look at what today's GPT 4o model can do, it makes more sense, because the context can be an image, i.e. pixels, a sequence of pixels, or video. So this network generalizes more and is universal. Just like a transformer. And above all, it is scalable, because from one neuron can be created small network. By adding layers or blocks, i.e. more neurons (parameters), we simply enlarge the network. So it is SCALABLE. In the same way as:

```
Single Transistor → Logic Gate → Integrated Circuit → Processor → Supercomputer
Single Bit → Byte → File → Operating System
Single Pixel → Image → Animation → Movie → Virtual Reality
Single Neuron → Neural Network → LLM
Atom → Molecule → Organism → Ecosystem
```

<b>The same principles are behind neural networks. Besides, the main idea was to create something that works like a BRAIN.</b>

<h3>3.</h3> Comparing this demo here to the current GPT 4o, I probably won't achieve what "4o" can do, because even looking at some of the commands I give to the model today and how it understands complex commands...  that there is something with RF (reinforcement learning) there. Because currently this model really surprises me sometimes, the previous 3 and 3.5 were not so "smart". <b>And above all, GPT is a serious project made by really smart people. This is serious really.</b></b>
<h3>4.</h3> Ok, to sum up. I just want to manually change the parameters here, and see what happens in the charts that you can see here on the bottom left in the image below. Which are generated from chart.js. The one at the top shows the parameters for the first layer (there are 2, I think). The one at the bottom is the probability distribution for letters, there are 27. You can change the parameters of both layers with the sliders, lower left corner. Something works... but not as it should yet. You can paste these names and make connections by pressing the "Train Examples" button under the textarea, but in this demo it has no effect on the network. In another demo I have it slightly improved. But here these visual graphs of layers and activation functions matter.
<br /><br />
That's pretty much it. I'll keep tweaking it because I want to get the same effect as here in generating names https://github.com/EurekaLabsAI/mlp/tree/master
<h2>How to run</h2>

Needs to install some libraries like Flask, etc.

```
from flask import Flask, request, jsonify, send_from_directory
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import math
```

And run via command

```
python serv.py
```

Works on localhost, port 5000

```
http://localhost:5000/
```

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/08-04-2025%20-%20EurekaLabs%20practice%20-%20update%20for%2025-03-2025%20demo/152%20-%2008-04-2025%20-%20i%20to%20ma%20juz%20sens.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/08-04-2025%20-%20EurekaLabs%20practice%20-%20update%20for%2025-03-2025%20demo/153%20-%2008-04-2025%20-%20cd.png?raw=true)

<hr>
Update 09-04-2025 - One thing.
<br /><br />
Update function in serv.py predict(), lines ~212-262. We call to calculate_prediction_loss_for_context


```
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
    
    loss_val, _xa, _xb = calculate_prediction_loss_for_context(input_tensor, predicted_letter)
    print( " LOSS : ", loss_val, " | ", _xa, " | ", _xb)
    
    # Remove hooks
    for handle in hook_handles:
        handle.remove()
    
    return jsonify({
        "input": input_sequence,
        "predicted": predicted_letter,
        "probabilities": probabilities,
        "activations": activations
    })
```

Then fix calculate_prediction_loss_for_context in serv.py, line ~307

```
def calculate_prediction_loss_for_context(context_seq, target_char):
    """
    Calculate cross-entropy loss for a single prediction given a context and a target.
    
    context_seq: list of characters (should be 3 tokens for a 3-token context)
    target_char: a single character string (the target token)
    """
    print("*"*80)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    
    print(context_seq, " ===> ", target_char)
    
    print("Type of context_seq[0]:", type(context_seq[0]))

    
    # Ensure the context is exactly model.context_length tokens.
    #idxs = [char_to_idx.get(ch, 0) for ch in context_seq]
    #idxs = context_seq[0][-1:] #[char_to_idx.get(ch, 0) for ch in context_seq]
    # Ensure we are working with a list of characters
    if isinstance(context_seq, torch.Tensor):
        context_seq = context_seq.squeeze().tolist()
        # Optional: convert idxs back to chars if needed
        context_seq = [idx_to_char.get(i, 'A') for i in context_seq]
        print("Converted tensor context_seq to characters:", context_seq)

    # Convert characters to indices
    idxs = [char_to_idx.get(ch, 0) for ch in context_seq]
    
    print(" IDXS ", idxs)
    while len(idxs) < model.context_length:
        idxs.insert(0, 0)  # pad with the index for 'A' (or zero) if needed
        
    input_tensor = torch.tensor([idxs])  # shape [1, context_length]
    target_idx = char_to_idx.get(target_char, 0)
    target_tensor = torch.tensor([target_idx])  # shape [1]
    
    print( " INPUT TENSOR : ", input_tensor)
    # Forward pass
    logits = model(input_tensor) + base_logits  # add interactive influence if desired
    # Compute loss
    #print("*"*80)
    print(logits)
    print(target_tensor)
    
    loss = loss_fn(logits, target_tensor)
    return loss.item(), logits, target_idx
```

And then we get "correct" calculation for loss. I write "correct" because for now it's just a skeleton of the application and it doesn't count at all according to the example that Andrej gave in the MLP code. Which is derived from forward and backward pass based on backward propagation. Then we get this IN CONSOLE:

```
********************************************************************************
tensor([[ 1, 13,  9]])  ===>  S
Type of context_seq[0]: <class 'torch.Tensor'>
Converted tensor context_seq to characters: ['B', 'N', 'J']
 IDXS  [1, 13, 9]
 INPUT TENSOR :  tensor([[ 1, 13,  9]])
tensor([[-1.9619e+00, -3.3757e-01, -2.3423e+00,  1.0635e+00,  5.8104e-01,
          1.5556e+00,  5.7750e-01,  3.6473e-01,  2.9283e-01,  3.1846e+00,
          6.0870e-01, -7.9169e-01,  9.5719e-01,  1.9237e+00, -6.4809e-01,
         -1.8184e+00, -1.0064e+00,  1.1055e-01,  4.0090e+00, -2.6134e+00,
          1.2316e+00,  2.9173e-01, -1.0378e+00, -8.5156e-01, -4.5651e+00,
         -3.8136e+00, -2.6055e-03]], grad_fn=<AddBackward0>)
tensor([18])
 LOSS :  0.733713686466217  |  tensor([[-1.9619e+00, -3.3757e-01, -2.3423e+00,  1.0635e+00,  5.8104e
-01,
          1.5556e+00,  5.7750e-01,  3.6473e-01,  2.9283e-01,  3.1846e+00,
          6.0870e-01, -7.9169e-01,  9.5719e-01,  1.9237e+00, -6.4809e-01,
         -1.8184e+00, -1.0064e+00,  1.1055e-01,  4.0090e+00, -2.6134e+00,
          1.2316e+00,  2.9173e-01, -1.0378e+00, -8.5156e-01, -4.5651e+00,
         -3.8136e+00, -2.6055e-03]], grad_fn=<AddBackward0>)  |  18
127.0.0.1 - - [09/Apr/2025 20:17:43] "POST /predict HTTP/1.1" 200 -
```

For now I need to fix and "fine-tune" it to generate names.
