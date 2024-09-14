05-09-2024 - The moment we stopped understanding AI [AlexNet] ((Welch Labs) - https://www.youtube.com/watch?v=UZDiGooFs54&ab_channel=WelchLabs <br />
Interesing wideo. And this is just the beginning, like the first computers. Year after year... but here we are, and we have the opportunity to learn from the best people on this field, like Andrej Karpathy. What will happen in next 10, 20, 30 years? I wonder what we will say then... ML, this is not my filed, but I like statistics and plots. Tesla autopilot, GPT 4 and other stuff from Open AI, Deep Mind / Google Brain etc show us, this is serious game! So.... ?
<br /><br />
Another great example video is in this channel on YT, and series videos about ML and LLM, <b>but this is important fact!</b> from ~13:14 - https://youtu.be/9-Jl0dxWQs8?t=794 -- "but if that neuron was inactive, if that number was zero, then this would have NO EFFECT".
<br /><br />
![dump](https://raw.githubusercontent.com/KarolDuracz/scratchpad/main/MachineLearning/ML%20with%20EurekaLabs/how%20might%20llms%20store%20facts%20chapter%207%203blue1brown%20--%20first.png)
![dump](https://raw.githubusercontent.com/KarolDuracz/scratchpad/main/MachineLearning/ML%20with%20EurekaLabs/how%20might%20llms%20store%20facts%20chapter%207%203blue1brown.png)
<br /><br />
<hr>
Maybe some useful stuff for future learning
<br /><br />
Untitled1.ipynb --> https://github.com/EurekaLabsAI/micrograd
<br /><br />
This file Untitled1.ipynb has some example code to compare with binary classification example from Andrej's micrograd EurekaLabs repo. 
<br /><br />
Untitled3.ipynb is for this track_simulator demos ==> https://github.com/KarolDuracz/scratchpad/tree/main/Webapp/Simple%20http%20server%20python3
<br />
There is some example calculation using chain rule to calculate some stuff for basic physics and game engine : tyre consumption against temperature,humidity and speed. But this is simple introduce for this issue. Like I wrote in that repo, here is a lot of things starting but not finalized. Once again, maybe someday...
<br /><br />
At this moment this is all I have and might be some interesting in future to continue learning<br />
--- 04-09-2024 --- 
<hr>
https://x.com/karpathy/status/1803963383018066272

```
These 94 lines of code are everything that is needed to train a neural network. Everything else is just efficiency.

This is my earlier project Micrograd. It implements a scalar-valued auto-grad engine. You start with some numbers at the leafs (usually the input data and the neural network parameters), build up a computational graph with operations like + and * that mix them, and the graph ends with a single value at the very end (the loss). You then go backwards through the graph applying chain rule at each node to calculate the gradients. The gradients tell you how to nudge your parameters to decrease the loss (and hence improve your network).

Sometimes when things get too complicated, I come back to this code and just breathe a little. But ok ok you also do have to know what the computational graph should be (e.g. MLP -> Transformer), what the loss function should be (e.g. autoregressive/diffusion), how to best use the gradients for a parameter update (e.g. SGD -> AdamW) etc etc. But it is the core of what is mostly happening.

The 1986 paper from Rumelhart, Hinton, Williams that popularized and used this algorithm (backpropagation) for training neural nets:
```

![dump](https://raw.githubusercontent.com/KarolDuracz/scratchpad/main/MachineLearning/ML%20with%20EurekaLabs/GQjvVdCakAEwVgD.jpg)

https://x.com/karpathy/status/1756380066580455557 
<br /><br />
Somewhere on the recordings about the history of the first successes of Microsoft and Windows I heard about work on Excel. At that time, from what I heard, 200-300 people worked on this project. At its beginnings. Could you do without this tool in everyday tasks? I don't know. But many things can be better organized thanks to it. Stories like this show that in the meantime, a lot of other interesting tools may emerge and become indispensable. But I still have in the back of my head this story that I once heard somewhere on TV about Excel. AI field in the next few years will probably create models and assistants that will be  better and better for daily tasks. And such virtual assistants from law, taxes, who have compressed knowledge are a breakthrough technology. And this change everything. Because the model, just like a human, has access to the same source of knowledge. etc etc etc. That's why LLM is interesting.
