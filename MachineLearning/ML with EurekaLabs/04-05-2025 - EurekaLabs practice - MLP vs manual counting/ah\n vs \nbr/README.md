<h2> combinations from TOP 30 "ah\n" -> 1 8 0 VS. "\nbr" -> 0 2 18</h2>
This chart comes from - https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/04-05-2025%20-%20EurekaLabs%20practice%20-%20MLP%20vs%20manual%20counting/Untitled8.ipynb

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/04-05-2025%20-%20EurekaLabs%20practice%20-%20MLP%20vs%20manual%20counting/ah%5Cn%20vs%20%5Cnbr/wykres%20test%20a%20vs%20test%20b.png?raw=true)

<h2>ah\n</h2>

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/04-05-2025%20-%20EurekaLabs%20practice%20-%20MLP%20vs%20manual%20counting/ah%5Cn%20vs%20%5Cnbr/test%20a.png?raw=true)

<h2>\nbr</h2>

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/04-05-2025%20-%20EurekaLabs%20practice%20-%20MLP%20vs%20manual%20counting/ah%5Cn%20vs%20%5Cnbr/test%20b.png?raw=true)

MLP code that displays this information

```
"""
Implements a simple n-gram language model in NumPy.
PyTorch has the Autograd engine, which calculates the gradients for us.
But in NumPy, we have to do the backward pass manually.
"""
import math
import numpy as np

from common import RNG, StepTimer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# NumPy implementation of a Multilayer Perceptron (MLP)

class MLP:
    """
    Takes the previous n tokens, encodes them with a lookup table,
    concatenates the vectors, and predicts the next token with an MLP.

    Reference:
    Bengio et al. 2003 https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
    """

    def __init__(self, rng, vocab_size, context_length, embedding_size, hidden_size):
        v, t, e, h = vocab_size, context_length, embedding_size, hidden_size
        self.embedding_size = embedding_size
        # initialize the weights the same way PyTorch does
        self.wte = np.asarray(rng.randn(v * e, mu=0, sigma=1.0), dtype=np.float32).reshape(v, e)
        scale = 1 / math.sqrt(e * t)
        self.fc1_weights = np.asarray(rng.rand(t * e * h, -scale, scale), dtype=np.float32).reshape(h, t * e).T
        self.fc1_bias = np.asarray(rng.rand(h, -scale, scale), dtype=np.float32)
        scale = 1 / math.sqrt(h)
        self.fc2_weights = np.asarray(rng.rand(h * v, -scale, scale), dtype=np.float32).reshape(v, h).T
        self.fc2_bias = np.asarray(rng.rand(v, -scale, scale), dtype=np.float32)
        # cache for the activations for the backward pass
        self.cache = {}

    def parameters(self):
        return {
            'wte': self.wte,
            'fc1_weights': self.fc1_weights,
            'fc1_bias': self.fc1_bias,
            'fc2_weights': self.fc2_weights,
            'fc2_bias': self.fc2_bias
        }

    def __call__(self, idx, targets=None):
        return self.forward(idx, targets)

    def forward(self, idx, targets=None):
        # idx are the input tokens, (B, T) numpy array of integers
        # targets are the target tokens, (B, ) tensor of integers
        B, T = idx.shape
        # encode all the tokens using the embedding table
        emb = self.wte[idx] # (B, T, embedding_size)
        # concat all of the embeddings together
        emb = emb.reshape(B, -1) # (B, T * embedding_size)
        # forward through the MLP
        hidden = np.tanh(emb @ self.fc1_weights + self.fc1_bias)
        logits = hidden @ self.fc2_weights + self.fc2_bias

        # cache some of the activations for the backward pass later
        self.cache['idx'] = idx
        self.cache['targets'] = targets
        self.cache['emb'] = emb
        self.cache['hidden'] = hidden

        # if we are given desired targets, also calculate the loss
        loss = None
        self.cache['probs'] = None
        if targets is not None:
            # cross-entropy loss, equivalent to F.cross_entropy in PyTorch
            logits_max = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            probs_targets = probs[np.arange(len(targets)), targets]
            nlls = -np.log(probs_targets)
            loss = np.mean(nlls)
            self.cache['probs'] = probs

        return logits, loss

    def backward(self):
        # extract the activations from the forward pass
        idx = self.cache['idx']
        targets = self.cache['targets']
        emb = self.cache['emb']
        hidden = self.cache['hidden']
        probs = self.cache['probs']
        B, T = idx.shape # batch, time

        # backward through the cross entropy loss
        dlogits = probs
        dlogits[np.arange(len(targets)), targets] -= 1
        dlogits /= len(targets)
        # backward through the last linear layer of the MLP
        dfc2_weights = hidden.T @ dlogits
        dfc2_bias = np.sum(dlogits, axis=0)
        dhidden = dlogits @ self.fc2_weights.T
        # backward through the tanh activation
        dprehidden = dhidden * (1 - hidden ** 2)
        # backward through the first linear layer of the MLP
        dfc1_weights = emb.T @ dprehidden
        dfc1_bias = np.sum(dprehidden, axis=0)
        demb = (dprehidden @ self.fc1_weights.T).reshape(B, T, self.embedding_size)
        # backward through the embedding table
        dwte = np.zeros_like(self.wte)
        # TODO: iirc there is a vectorized way to do this
        for i in range(B):
            for j in range(T):
                dwte[idx[i, j]] += demb[i, j]

        return {
            'wte': dwte,
            'fc1_weights': dfc1_weights,
            'fc1_bias': dfc1_bias,
            'fc2_weights': dfc2_weights,
            'fc2_bias': dfc2_bias
        }

# -----------------------------------------------------------------------------
# AdamW optimizer

class AdamW:

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, weight_decay=1e-4, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def set_lr(self, lr):
        self.lr = lr

    def step(self, grads):
        self.t += 1
        for k in self.params.keys():
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * grads[k]**2
            m_hat = self.m[k] / (1 - self.beta1**self.t)
            v_hat = self.v[k] / (1 - self.beta2**self.t)
            self.params[k] -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * self.params[k])

# -----------------------------------------------------------------------------
# simple DataLoader that iterates over all the n-grams

def dataloader(tokens, context_length, batch_size):
    # returns inputs, targets as torch Tensors of shape (B, T), (B, )
    n = len(tokens)
    inputs, targets = [], []
    pos = 0
    while True:
        # simple sliding window over the tokens, of size context_length + 1
        window = tokens[pos:pos + context_length + 1]
        inputs.append(window[:-1])
        targets.append(window[-1])
        # once we've collected a batch, emit it
        if len(inputs) == batch_size:
            yield (np.array(inputs), np.array(targets))
            inputs, targets = [], []
        # advance the position and wrap around if we reach the end
        pos += 1
        if pos + context_length >= n:
            pos = 0

# -----------------------------------------------------------------------------
# evaluation function

def eval_split(model, tokens, max_batches=None):
    # calculate the loss on the given tokens
    total_loss = 0
    num_batches = len(tokens) // batch_size
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)
    data_iter = dataloader(tokens, context_length, batch_size)
    for _ in range(num_batches):
        inputs, targets = next(data_iter)
        logits, loss = model(inputs, targets)
        total_loss += loss.item()
    mean_loss = total_loss / num_batches
    return mean_loss

# -----------------------------------------------------------------------------
# sampling form the model

def softmax(logits):
    # logits here is a (1D) numpy.array of shape (V,)
    maxval = np.max(logits) # subtract max for numerical stability
    exps = np.exp(logits - maxval)
    probs = exps / np.sum(exps)
    return probs

def sample_discrete(probs, coinf):
    # sample from a discrete distribution
    # probs is a 1D numpy array of shape (V,)
    cdf = 0.0
    for i, prob in enumerate(probs):
        cdf += prob
        if coinf < cdf:
            return i
    return len(probs) - 1  # in case of rounding errors

# -----------------------------------------------------------------------------
# let's train!

# "train" the Tokenizer, so we're able to map between characters and tokens
train_text = open('data/train.txt', 'r').read()
assert all(c == '\n' or ('a' <= c <= 'z') for c in train_text)
uchars = sorted(list(set(train_text))) # unique characters we see in the input
vocab_size = len(uchars)
char_to_token = {c: i for i, c in enumerate(uchars)}
token_to_char = {i: c for i, c in enumerate(uchars)}
EOT_TOKEN = char_to_token['\n'] # designate \n as the delimiting <|endoftext|> token
# pre-tokenize all the splits one time up here
test_tokens = [char_to_token[c] for c in open('data/test.txt', 'r').read()]
val_tokens = [char_to_token[c] for c in open('data/val.txt', 'r').read()]
train_tokens = [char_to_token[c] for c in open('data/train.txt', 'r').read()]


# create the model
context_length = 3 # if 3 tokens predict the 4th, this is a 4-gram model
embedding_size = 48
hidden_size = 512
init_rng = RNG(1337)
model = MLP(init_rng, vocab_size, context_length, embedding_size, hidden_size)

def count_model_parameters(params):
    total = 0
    for name, param in params.items():
        shape = param.shape
        param_count = 1
        for dim in shape:
            param_count *= dim
        print(f"{name} ==> {shape} | Parameters: {param_count}")
        total += param_count
    print(f"\nTotal Parameters: {total}")
    return total

for key in model.parameters().items():
    print(key[0], " ==>", key[1].shape)

print(" == total parameters == ")
count_model_parameters(model.parameters())

# optimizer
learning_rate = 7e-4
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# training loop
timer = StepTimer()
batch_size = 128
num_steps = 50000 # default 50k -> 50000
print(f'num_steps {num_steps}, num_epochs {num_steps * batch_size / len(train_tokens):.2f}')
train_data_iter = dataloader(train_tokens, context_length, batch_size)

counter_for_window = 0
counter_for_window_neg = 0
n_loops = 0
# ah\n
#print(char_to_token['a'], " ", char_to_token['h'], " ", char_to_token['\n'])
# \nbr
print(char_to_token['\n'], " ", char_to_token['b'], " ", char_to_token['r'])
print(80*'-')
__arr_step = []
__arr_cycle_in_batch = []

for step in range(num_steps):
    # cosine learning rate schedule, from max lr to 0
    lr = learning_rate * 0.5 * (1 + math.cos(math.pi * step / num_steps))
    optimizer.set_lr(lr)
    # every now and then evaluate the validation loss
    last_step = step == num_steps - 1
    if step % 200 == 0 or last_step:
        train_loss = eval_split(model, train_tokens, max_batches=20)
        val_loss = eval_split(model, val_tokens)
        print(f'step {step:6d} | train_loss {train_loss:.6f} | val_loss {val_loss:.6f} | lr {lr:e} | time/step {timer.get_dt()*1000:.4f}ms')
        print(counter_for_window, " ", counter_for_window_neg)
    # training step
    #n_loops += 1
    with timer:
        # get the next batch of training data

        inputs, targets = next(train_data_iter)
        
        for ij in range(batch_size):
            
            """
            n_test = ((torch.tensor(inputs[ij]).eq(torch.tensor([18,1,25]))))
            bool_test = torch.all(n_test)
            print(" is eq with counter_for_window : ", ((torch.tensor(inputs[ij]).eq(torch.tensor([18,1,25])))), " ==> ", bool_test, " check: ", (bool_test==True), " val: ", counter_for_window)
            if bool_test==True:
                counter_for_window += 1   
            """
            
            n_test = ((torch.tensor(inputs[ij]).eq(torch.tensor([1,8,0])))) # i'm lookin for "ah\n" -> 1 8 0
            #n_test = ((torch.tensor(inputs[ij]).eq(torch.tensor([0,2,18])))) # i'm lookin for "\nbr" -> 0 2 18
            bool_test = torch.all(n_test)
            #print(bool_test, "  ", bool_test.item(), " ", (bool_test.item()==True), " ==>", n_test)
            if (bool_test.item() == True):
                #print("positive ", torch.tensor(inputs[ij]), " index : ", ij, " step: ", step)
                counter_for_window += 1
                __arr_step.append(step)
                __arr_cycle_in_batch.append(ij)
            else:
                #print("negative ", torch.tensor(inputs[ij]))
                counter_for_window_neg += 1
            #print(" is eq with counter_for_window : ", ((torch.tensor(inputs[ij]).eq(torch.tensor([18,1,25])))), " ==> ", bool_test, " check: ", (bool_test==True), " val: ", counter_for_window)
            #if (bool_test.item()==True)==True:
                #print(n_test)
                #print(step, " -> ",  counter_for_window)
                #counter_for_window += 1
            
            
   
        
        """
        print(inputs.shape)
        print(inputs)   
        print(inputs[0].shape)
        print(inputs[0], " | ", torch.tensor([18,1,25]))  
        print(inputs[0].shape, " ",  torch.tensor([18,1,25]).shape )
        print(torch.tensor(inputs[0]).shape, "-> ", torch.tensor(inputs[0]).shape[0])
        print(torch.tensor(inputs[0])[0].item())
        ####
        for g1 in range(torch.tensor(inputs[0]).shape[0]):
            print(torch.tensor(inputs[0])[g1].item())
        print(" is eq : ", ((torch.tensor(inputs[0]).eq(torch.tensor([18,1,25])))))
        print(" is eq with counter_for_window : ", ((torch.tensor(inputs[counter_for_window]).eq(torch.tensor([18,1,25])))))
        print(" window counter ", counter_for_window)
        #print(token_to_char([inputs[0][0]]))
        counter_for_window += 1
        if counter_for_window == batch_size:
            counter_for_window = 0
        """
        
        # forward through the model
        logits, loss = model(inputs, targets)
        # backpropagate and update the weights
        grads = model.backward()
        # step the optimizer
        optimizer.step(grads)

# model inference
# hardcode a prompt from which we'll continue the text
sample_rng = RNG(42)
prompt = "\nrichard"
context = [char_to_token[c] for c in prompt]
assert len(context) >= context_length
context = context[-context_length:] # crop to context_length
print(prompt, end='', flush=True)
# now let's sample 200 more tokens that follow
for _ in range(1000):
    # take the last context_length tokens and predict the next one
    context_array = np.array(context).reshape(1, -1) # (1, T)
    logits, _ = model(context_array) # (1, V)
    probs = softmax(logits[0]) # (V, )
    #print(probs)
    coinf = sample_rng.random() # "coin flip", float32 in range [0, 1)
    next_token = sample_discrete(probs, coinf)
    context = context[1:] + [next_token] # update the token tape
    
    print(token_to_char[next_token], end='', flush=True)
print() # newline

# and finally report the test loss
test_loss = eval_split(model, test_tokens)
print(f'test_loss {test_loss:.6f}')
```
