<h2>Still TODO</h2>
Today's post is just to give an idea of ​​what I wrote in my post here
https://github.com/KarolDuracz/scratchpad/tree/main/MachineLearning/ML%20with%20EurekaLabs/25-03-2025%20-%20EurekaLabs%20practice
<br /><br />
Still nothing works here properly. But I'll finish it because it helps me learn a bit about how the network works from the inside, even just trying to do something similar to the MLP with context_length = 3. <b>I will try to make a working model in the next few weeks to close the topic. </b> <br /><br />
1. While playing around with ChatGPT 4o I also realized a few things, that what I came up with here is a graph approach to networks, i.e. "graph network".<br /><br />
2. Even if I manage to finish it and make it predict exactly like Andrej's MLP that generates names learned from this dataset  https://github.com/EurekaLabsAI/mlp/blob/master/data/train.txt . And when I then add a larger tokenizer like "GPT3.5 base" it probably has nothing to do with how the problem is generalized by neural networks and the backpropagation algorithm, which changes parameters in small steps to reduce the loss function. In general, when I look at what today's GPT 4o model can do, it makes more sense, because the context can be an image, i.e. pixels, a sequence of pixels, or video. So this network generalizes more and is universal. Just like a transformer. And above all, it is scalable, because from 1 neuron a network of parameters and connections is created that predict something at the end. And it is enough to add more parameters, so it is SCALABLE.<br /><br />
3. Comparing this demo here to the current GPT 4o, I probably won't achieve what "4o" can do, because even looking at some of the commands I give to the model today and how it understands complex commands...  that there is something with RF (reinforcement learning) there. Because currently this model really surprises me sometimes, the previous 3 and 3.5 were not so "smart". <b>And above all, GPT is a serious project made by really smart people. This is serious really.</b></b><br /><br />
4. Ok, to sum up. I just want to manually change the parameters here, and see what happens in the charts that you can see here on the bottom left in the image below. Which are generated from chart.js. The one at the top shows the parameters for the first layer (there are 2, I think). The one at the bottom is the probability distribution for letters, there are 27. You can change the parameters of both layers with the sliders, lower left corner. Something works... but not as it should yet. You can paste these names and make connections by pressing the "Train Examples" button under the textarea, but in this demo it has no effect on the network. In another demo I have it slightly improved. But here these visual graphs of layers and activation functions matter.
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

