<h2>Update 02-06-2025 - I do not recommend reading everything. Most of it is my personal notes, thoughts. FOR THE VISITOR THIS HAS NO POSITIVE VALUE. IT IS ONLY WORTH START WITH THIS POST.</h2>
1. My notes and a short analysis of Andrej's first MLP videos https://github.com/KarolDuracz/scratchpad/tree/main/MachineLearning/ML%20with%20EurekaLabs/04-05-2025%20-%20EurekaLabs%20practice%20-%20MLP%20vs%20manual%20counting <br />
2. The first 2 paragraphs are a continuation of the 04-05-2025 demo, the rest is a continuation of my notes and plans. Nothing important beyond the first 2 paragraphs on "The EM Algorithm Clearly Explained (Expectation-Maximization Algorithm" https://github.com/KarolDuracz/scratchpad/tree/main/MachineLearning/ML%20with%20EurekaLabs/11-05-2025%20-%20EurekaLabs%20practice%20-%20MNIST
<br /><br />
THE REST ARE MY NOTES ON LEARNING ML. I do not recommend reading this thoroughly.
<hr>
<br /><br />
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

<hr>
Update 7-11-2024 - I forgot about this post by Andrej. http://karpathy.github.io/neuralnets/ <br />
And after this post neuralnets + Andrej's first video about micrograd I understood how backpropagation works. <br />
Everyone should read this post after the micrograd video IMHO.
There is also a link to https://cs231n.github.io/ in this post. I haven't found the topic of backpropagation explained better in such a simple and understandable way as here.

<br />
<hr>
And ofcourse this series, and CS231n Winter 2016: Lecture 5: Neural Networks Part 2 https://www.youtube.com/watch?v=gYpoJMlgyXA&ab_channel=AndrejKarpathy . I saw this video some time ago,  were "makemore" series came out etc. on his new YT channel. But back then he had a really good mindset and he has able to explain these topics in detail in an understandable way. The new Andrej's channel describes in detail how it works and how to implement it.
