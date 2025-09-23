In theory, one neuron has inputs and outputs. And that's all it takes to make a decision. To calculate the next letter, word, token, etc. It's just math. This is a big simplification of how really it is and how its compute probabilities to make a decisions, maybe too big, but to have some point of reference. 

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/23-09-2025%20-%20MLP%20vs%20INDEX%20for%20words%20approach/_perceptron.png?raw=true)

<br />
Just for comparison with MLP. 

<br /><br />
What if we built an index of words and only calculated the prediction error for letters? For example, if the index under the letter "l" (lowercase l) contains possible combinations of "log","loss","let","length," then knowing the subsequent letters, we could calculate the error for the remainder and eliminate other possibilities (thus selecting the correct word) by writing subsequent letters. How do you build such a word index? The typical approach is to take the first letter of the word. And based on subsequent letters, we reduce the number of possible combinations, which is clearly visible in the HTML demo.
<br /><br />
<b>The model training doesn't work optimally. This is just a demo for comparison.</b>

notebook with python code built on the pytorch Library - The first block, after training, simulates typing the words "log, loss" in few steps, i.e., char by char and probability distribution. The second listing, after training, allows us to type subsequent letters in the console and see possible word combinations.

https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/23-09-2025%20-%20MLP%20vs%20INDEX%20for%20words%20approach/MLP%20vs%20INDEX%20for%20words%20approach.ipynb

HTML interactive app ( no needs any serwer, it's just  runs on web browser ) - THE MODEL DOESN'T TRAIN WELL, BUT ON THIS WORD INDEX IT IS ENOUGH FOR IT TO PREDICT WORDS RATHER CORRECTLY. This is not a 100% working implementation like the one shown by Andrej in his Ngram or MLP training approach -> https://github.com/EurekaLabsAI 

https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/23-09-2025%20-%20MLP%20vs%20INDEX%20for%20words%20approach/demo_app.html

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/23-09-2025%20-%20MLP%20vs%20INDEX%20for%20words%20approach/screen%20for%20demo%2023-09-2025.png?raw=true)

Add more keywords, e.g. for JavaScript to HTML app ( line ~75 )

```
const WORDS = [
  // ECMAScript keywords & control
  "break","case","catch","class","const","continue","debugger","default","delete","do",
  "else","export","extends","finally","for","function","if","import","in","instanceof",
  "let","new","return","super","switch","this","throw","try","typeof","var","void",
  "while","with","yield","await","async","static","get","set","of",

  // common globals / constructors / types
  "console","math","number","string","boolean","array","object","json","date","promise",
  "map","set","weakmap","weakset","symbol","regexp","error","eval",

  // Promise / async helpers
  "then","catch","resolve","reject","async","await","finally",

  // Node / module-ish
  "module","exports","require","__dirname","__filename",

  // Common runtime & util functions
  "parseint","parsefloat","isnan","isfinite","encodeuri","decodeduri","encodeuricomponent",
  "decodeuricomponent","jsonstringify","jsonparse",

  // Math
  "floor","ceil","round","random","abs","min","max","pow","sqrt","log","exp",

  // common console helpers
  "log","warn","error","info","debug","table","time","timeend",

  // browser globals / DOM
  "window","document","location","history","navigator","localstorage","sessionstorage",
  "alert","confirm","prompt","fetch","addeventlistener","removeeventlistener",
  "queryselector","queryselectorall","getelementbyid","getelementsbyclassname","getelementsbytagname",
  "createelement","appendchild","removechild","replacechild","classlist","classname","dataset",
  "innerhtml","textcontent","value","style",

  // timers
  "settimeout","cleartimeout","setinterval","clearinterval",

  // Array methods
  "push","pop","shift","unshift","splice","slice","map","filter","reduce","foreach",
  "find","findindex","includes","indexof","join","split","concat","sort","reverse",

  // String methods
  "replace","tolowercase","touppercase","trim","substr","substring","startswith","endswith",
  "indexof","includes","split","concat",

  // common object / prototype helpers
  "prototype","constructor","hasownproperty","keys","values","entries","assign","create","defineproperty",

  // common DOM / UI helpers & patterns
  "addclass","removeclass","toggleclass","append","prepend","closest","matches","contains",

  // common libs / patterns (generic)
  "jquery","react","vue","angular","redux","rxjs",

  // misc identifiers often seen in JS code
  "id","name","type","length","size","index","key","value","payload","props","state","setstate",
  "dispatch","subscribe","unsubscribe","handler","callback","errorhandler","response","request"
].map(s => s.toLowerCase());
```

demo to show how many words there are under a given letter - https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/23-09-2025%20-%20MLP%20vs%20INDEX%20for%20words%20approach/word_index.html

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/23-09-2025%20-%20MLP%20vs%20INDEX%20for%20words%20approach/word%20index.png?raw=true)

Training the model for the larger "WORDS" array in the HTML demo takes a bit longer. For example, if the word "float" isn't present, the model tries to predict anything anyway.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/23-09-2025%20-%20MLP%20vs%20INDEX%20for%20words%20approach/extended%20WORDS%20table.png?raw=true)

But what interests me most about this approach is the elimination of other possibilities. That is, the more letters there are, the smaller the range of possibilities. It may not have given the correct prediction for "float", but the most accurate words are those with the first 3 letters "flo", the rest are eliminated because their "probs" value is too low. After [05:18:39] Epoch 53/60 — avg batch loss 1.3713 floor is higher than the filter for "float".

```
It may not work optimally, but it works to some extent. The letter "c" in the second larger WORDS
array provides a good example of how the system works ( there are more starting with the letter s ).
Checking the word count in word_index.html, for the letter "c" there are 19 words: c (19)
callback
case
catch
ceil
class
classlist
classname
clearinterval
cleartimeout
closest
concat
confirm
console
const
constructor
contains
continue
create
createelement

And for just one letter, "c," this HTML demo shows 60 steps:
word prob
concat 0.1399
constructor 0.1324
catch 0.1271
ceil 0.0916
create 0.0560
continue 0.0546
case 0.0545
class 0.0542

For two letters, for example, "cl," there are 6 possibilities, then "ca" only 3, and for "co" as many as 7.
How should the system work? Example for "co"

word prob
constructor 0.5815
concat 0.1381
const 0.0807
console 0.0740
continue 0.0662
contains 0.0345
confirm 0.0225
closest 0.0025
```

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/23-09-2025%20-%20MLP%20vs%20INDEX%20for%20words%20approach/co%20example.png?raw=true)

In python + torch version it's slightly better and faster
<br /><br />
code

```
# interactive_index_predictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import defaultdict

# ---------------------------
# 1) Small word list + index
# ---------------------------
#WORDS = ["log", "loss", "let", "length", "const", "class",
#         "return", "function", "console", "float", "for", "if", "else"]

WORDS = [
  # ECMAScript keywords & control
  "break","case","catch","class","const","continue","debugger","default","delete","do",
  "else","export","extends","finally","for","function","if","import","in","instanceof",
  "let","new","return","super","switch","this","throw","try","typeof","var","void",
  "while","with","yield","await","async","static","get","set","of",

  # common globals / constructors / types
  "console","math","number","string","boolean","array","object","json","date","promise",
  "map","set","weakmap","weakset","symbol","regexp","error","eval",

  # Promise / async helpers
  "then","catch","resolve","reject","async","await","finally",

  # Node / module-ish
  "module","exports","require","__dirname","__filename",

  # Common runtime & util functions
  "parseint","parsefloat","isnan","isfinite","encodeuri","decodeduri","encodeuricomponent",
  "decodeuricomponent","jsonstringify","jsonparse",

  # Math
  "floor","ceil","round","random","abs","min","max","pow","sqrt","log","exp",

  # common console helpers
  "log","warn","error","info","debug","table","time","timeend",

  # browser globals / DOM
  "window","document","location","history","navigator","localstorage","sessionstorage",
  "alert","confirm","prompt","fetch","addeventlistener","removeeventlistener",
  "queryselector","queryselectorall","getelementbyid","getelementsbyclassname","getelementsbytagname",
  "createelement","appendchild","removechild","replacechild","classlist","classname","dataset",
  "innerhtml","textcontent","value","style",

   # timers
  "settimeout","cleartimeout","setinterval","clearinterval",

  # Array methods
  "push","pop","shift","unshift","splice","slice","map","filter","reduce","foreach",
  "find","findindex","includes","indexof","join","split","concat","sort","reverse",

  # String methods
  "replace","tolowercase","touppercase","trim","substr","substring","startswith","endswith",
  "indexof","includes","split","concat",

  # common object / prototype helpers
  "prototype","constructor","hasownproperty","keys","values","entries","assign","create","defineproperty",

  # common DOM / UI helpers & patterns
  "addclass","removeclass","toggleclass","append","prepend","closest","matches","contains",

  # common libs / patterns (generic)
  "jquery","react","vue","angular","redux","rxjs",

  # misc identifiers often seen in JS code
  "id","name","type","length","size","index","key","value","payload","props","state","setstate",
  "dispatch","subscribe","unsubscribe","handler","callback","errorhandler","response","request"
]

WORDS = [w.lower() for w in WORDS]
WORD2IDX = {w: i for i, w in enumerate(WORDS)}
IDX2WORD = {i: w for w, i in WORD2IDX.items()}

INDEX = defaultdict(list)
for w, idx in WORD2IDX.items():
    INDEX[w[0]].append(idx)

# ---------------------------
# 2) char vocabulary + padding
# ---------------------------
PAD = "<PAD>"
CHARS = sorted({c for w in WORDS for c in w})
CHARS = [PAD] + CHARS
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}
NUM_CHARS = len(CHARS)
NUM_WORDS = len(WORDS)
MAX_PREFIX = max(1, max(len(w) - 1 for w in WORDS))

# ---------------------------
# 3) dataset: prefixes -> word
# ---------------------------
examples = []
for w in WORDS:
    wi = WORD2IDX[w]
    # prefixes length 1..len(w)-1
    for L in range(1, len(w)):
        prefix = w[:L]
        idxs = [CHAR2IDX[ch] for ch in prefix]
        pad = [CHAR2IDX[PAD]] * (MAX_PREFIX - len(idxs))
        examples.append((idxs + pad, len(prefix), wi))
random.shuffle(examples)

Xs = torch.tensor([ex[0] for ex in examples], dtype=torch.long)
prefix_lens = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
Ys = torch.tensor([ex[2] for ex in examples], dtype=torch.long)

# ---------------------------
# 4) tiny model
# ---------------------------
class PrefixWordPredictor(nn.Module):
    def __init__(self, num_chars, emb_dim, hidden_dim, num_words):
        super().__init__()
        self.emb = nn.Embedding(num_chars, emb_dim, padding_idx=0)
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_words)

    def forward(self, prefix_idxs, prefix_lens, allowed_mask):
        emb = self.emb(prefix_idxs)                   # (B, L, E)
        valid = (prefix_idxs != 0).float().unsqueeze(-1)
        summed = (emb * valid).sum(dim=1)
        lengths = valid.sum(dim=1).clamp(min=1)
        pooled = summed / lengths
        h = F.relu(self.fc1(pooled))
        logits = self.fc2(h)
        very_neg = -1e9
        logits = logits + (~allowed_mask).float() * very_neg
        return logits

device = torch.device("cpu")
model = PrefixWordPredictor(NUM_CHARS, emb_dim=16, hidden_dim=32, num_words=NUM_WORDS).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.02)
loss_fn = nn.CrossEntropyLoss()

# quick training (tiny)
BATCH = 8
EPOCHS = 120
for epoch in range(EPOCHS):
    perm = torch.randperm(len(Xs))
    total = 0.0
    for i in range(0, len(Xs), BATCH):
        batch_idx = perm[i:i+BATCH]
        xb = Xs[batch_idx].to(device)
        yb = Ys[batch_idx].to(device)
        # build allowed_mask using first letter of each prefix (we know prefix length >=1)
        prefix_strings = []
        for bi in batch_idx:
            raw = Xs[bi].tolist()
            s = ''.join([IDX2CHAR[c] for c in raw if IDX2CHAR[c] != PAD])
            prefix_strings.append(s)
        masks = []
        for s in prefix_strings:
            m = torch.zeros(NUM_WORDS, dtype=torch.bool)
            m[INDEX[s[0]]] = True
            masks.append(m)
        allowed_mask = torch.stack(masks).to(device)
        logits = model(xb, prefix_lens[batch_idx].to(device), allowed_mask)
        loss = loss_fn(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * xb.size(0)
    if (epoch+1) % 40 == 0 or epoch == 0:
        print(f"[train] epoch {epoch+1}/{EPOCHS} avg loss {total/len(Xs):.4f}")

print("Training done.\n")

# ---------------------------
# 5) helper functions
# ---------------------------
def encode_prefix(prefix: str):
    idxs = [CHAR2IDX.get(c, 0) for c in prefix]
    pad = [CHAR2IDX[PAD]] * (MAX_PREFIX - len(idxs))
    return idxs + pad

def make_allowed_mask(prefix_list):
    masks = []
    for pre in prefix_list:
        m = torch.zeros(NUM_WORDS, dtype=torch.bool)
        if len(pre) == 0:
            m[:] = True  # allow all if empty (but our app requires >=1)
        else:
            m[INDEX[pre[0]]] = True
        masks.append(m)
    return torch.stack(masks, dim=0)

def pretty_print_preds(logits, topk=6):
    probs = F.softmax(logits, dim=-1).squeeze(0).detach().cpu()
    topv, topi = probs.topk(min(topk, probs.size(0)))
    out = [(IDX2WORD[i.item()], v.item()) for v,i in zip(topv, topi)]
    return out, probs

# ---------------------------
# 6) interactive loop
# ---------------------------
def interactive_loop():
    print("Interactive index-first predictor.")
    print("Words in index:", WORDS)
    print("Type single or multiple letters (prefix). Type 'quit' to exit.")
    print("Optionally enter a ground-truth target word (must be in the word list) to compute errors.")
    target = input("Enter ground-truth target word (or press Enter to skip): ").strip().lower()
    if target == "":
        target = None
    elif target not in WORD2IDX:
        print("Target not in word list. Ignoring.")
        target = None
    print("---- start typing prefixes ----")
    while True:
        pre = input("prefix> ").strip().lower()
        if pre == "quit":
            break
        if len(pre) == 0:
            print("Please type at least the first letter.")
            continue
        if pre[0] not in INDEX:
            print(f"No words starting with '{pre[0]}'.")
            continue
        enc = torch.tensor([encode_prefix(pre)], dtype=torch.long).to(device)
        allowed = make_allowed_mask([pre]).to(device)
        logits = model(enc, torch.tensor([len(pre)]), allowed)
        preds, probs = pretty_print_preds(logits, topk=6)
        allowed_indices = INDEX[pre[0]]
        allowed_mass = probs[allowed_indices].sum().item()
        print(f"\nTop candidates for prefix '{pre}':")
        for w, p in preds:
            print(f"  {w:12s}  p = {p:.3f}")
        print(f"Allowed-bucket (first-letter='{pre[0]}') total mass: {allowed_mass:.3f}")
        if target:
            true_i = WORD2IDX[target]
            p_true = probs[true_i].item()
            nll = -torch.log(torch.clamp(probs[true_i], 1e-9)).item()
            index_error = 1.0 - (p_true / allowed_mass) if allowed_mass > 0 else 1.0
            rank = (probs > probs[true_i]).sum().item() + 1
            print(f"Metrics (true = '{target}'):")
            print(f"  p(true) = {p_true:.4f}, NLL = {nll:.3f}, rank = {rank}, index_error = {index_error:.3f}")
        print("\n---")

if __name__ == "__main__":
    interactive_loop()
```

output log for, c > co > con

```
[train] epoch 1/120 avg loss 2.3601
[train] epoch 40/120 avg loss 1.2068
[train] epoch 80/120 avg loss 1.0907
[train] epoch 120/120 avg loss 1.0769
Training done.

Interactive index-first predictor.
Words in index: ['break', 'case', 'catch', 'class', 'const', 'continue', 'debugger', 'default', 'delete', 'do', 'else', 'export', 'extends', 'finally', 'for', 'function', 'if', 'import', 'in', 'instanceof', 'let', 'new', 'return', 'super', 'switch', 'this', 'throw', 'try', 'typeof', 'var', 'void', 'while', 'with', 'yield', 'await', 'async', 'static', 'get', 'set', 'of', 'console', 'math', 'number', 'string', 'boolean', 'array', 'object', 'json', 'date', 'promise', 'map', 'set', 'weakmap', 'weakset', 'symbol', 'regexp', 'error', 'eval', 'then', 'catch', 'resolve', 'reject', 'async', 'await', 'finally', 'module', 'exports', 'require', '__dirname', '__filename', 'parseint', 'parsefloat', 'isnan', 'isfinite', 'encodeuri', 'decodeduri', 'encodeuricomponent', 'decodeuricomponent', 'jsonstringify', 'jsonparse', 'floor', 'ceil', 'round', 'random', 'abs', 'min', 'max', 'pow', 'sqrt', 'log', 'exp', 'log', 'warn', 'error', 'info', 'debug', 'table', 'time', 'timeend', 'window', 'document', 'location', 'history', 'navigator', 'localstorage', 'sessionstorage', 'alert', 'confirm', 'prompt', 'fetch', 'addeventlistener', 'removeeventlistener', 'queryselector', 'queryselectorall', 'getelementbyid', 'getelementsbyclassname', 'getelementsbytagname', 'createelement', 'appendchild', 'removechild', 'replacechild', 'classlist', 'classname', 'dataset', 'innerhtml', 'textcontent', 'value', 'style', 'settimeout', 'cleartimeout', 'setinterval', 'clearinterval', 'push', 'pop', 'shift', 'unshift', 'splice', 'slice', 'map', 'filter', 'reduce', 'foreach', 'find', 'findindex', 'includes', 'indexof', 'join', 'split', 'concat', 'sort', 'reverse', 'replace', 'tolowercase', 'touppercase', 'trim', 'substr', 'substring', 'startswith', 'endswith', 'indexof', 'includes', 'split', 'concat', 'prototype', 'constructor', 'hasownproperty', 'keys', 'values', 'entries', 'assign', 'create', 'defineproperty', 'addclass', 'removeclass', 'toggleclass', 'append', 'prepend', 'closest', 'matches', 'contains', 'jquery', 'react', 'vue', 'angular', 'redux', 'rxjs', 'id', 'name', 'type', 'length', 'size', 'index', 'key', 'value', 'payload', 'props', 'state', 'setstate', 'dispatch', 'subscribe', 'unsubscribe', 'handler', 'callback', 'errorhandler', 'response', 'request']
Type single or multiple letters (prefix). Type 'quit' to exit.
Optionally enter a ground-truth target word (must be in the word list) to compute errors.
Enter ground-truth target word (or press Enter to skip):  co
Target not in word list. Ignoring.
---- start typing prefixes ----
prefix>  c

Top candidates for prefix 'c':
  concat        p = 0.105
  catch         p = 0.082
  confirm       p = 0.064
  clearinterval  p = 0.059
  cleartimeout  p = 0.059
  classname     p = 0.058
Allowed-bucket (first-letter='c') total mass: 1.000

---
prefix>  co

Top candidates for prefix 'co':
  console       p = 0.539
  concat        p = 0.201
  constructor   p = 0.094
  const         p = 0.065
  contains      p = 0.061
  confirm       p = 0.024
Allowed-bucket (first-letter='c') total mass: 1.000

---
prefix>  con

Top candidates for prefix 'con':
  console       p = 0.522
  concat        p = 0.133
  continue      p = 0.121
  contains      p = 0.112
  confirm       p = 0.069
  const         p = 0.036
Allowed-bucket (first-letter='c') total mass: 1.000
```

<h2>Demo app</h2>

https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/23-09-2025%20-%20MLP%20vs%20INDEX%20for%20words%20approach/app.py
<br /><br />
A demo using FLASK and sqlite3. Very simple. This is purely for testing purposes, to see if it makes sense or if it's a waste of time to develop it further. Just the base demo code, which might be useful with some tweaks. <b>Perhaps it would make sense to add a better prediction system and then measure the delta between what the user is typing and what system are predict and put in text to increase encoding speed. But this requires more complex structures, not just single words. To improve reaction time and visibility of available combinations ?!</b> For now, I'm leaving it to think about in free time.
<br /><br />
You need the FLASK library https://pypi.org/project/Flask/ (as in the previous demos I've added here. I personally like it for small demos).
<br /><br />
Everything is in one file: app.py: HTML, JS, and Python code.
<br /><br />
There are other tabs with /debug_error and time_counter_stats. But they only work superficially.
<br /><br />
RUN

```
python app.py
```

The server runs on localhost:5000 by default. After starting, simply start typing, for example, the letter "f" in the text field to see how the system works.
<br /><br />
<b>System does not use neural networks to calculate probabilities, this is purely for demonstration purposes for a simple INITIAL_WORDS array around line 13 in app.py</b>
<br /><br />
A short description is in the image.
<br /><br />
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/23-09-2025%20-%20MLP%20vs%20INDEX%20for%20words%20approach/250%20-%2023-09-2025%20-%20app%20screenshot.png?raw=true)
