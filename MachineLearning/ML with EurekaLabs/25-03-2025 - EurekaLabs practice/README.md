There are a few things I'd like to learn on small models like (https://github.com/EurekaLabsAI/mlp) before I get to grips with transformer and more complicated stuff like 
positional encoding, attention layer etc etc. Because this is one of the key mechanisms on which the architecture bases decisions 
about text generation. I'm only just now looking into how it works, day after day I try to understand it, so... 
<br /><br />
Ok. it's a job Chat GPT in a few prompts. It's pretty damn good. Understands the problem and knows what answer to give. But is it possible 
to manually arrange these tokens in a row, i.e. create a context like in the picture below to change the combinations? Or "draw" 
connections instead of training. Yes, it is not "machine learning" in a way with BACKPROP and math way of backpropagation to find "sweet spot" by follow loss function, but I think it can be an interesting task and such a dynamic 
change of network parameters by drawing a few connections, or typing in a few combinations of letters instead of training the entire 
data set... hmm...
<h2>TODO</h2>
Needs to finish it in general so that it works as described and after setting a few series of letters and entering these combinations into memory, then making a few connections, 
the model gives some predictions, like MLP (https://github.com/EurekaLabsAI/mlp) which is now trained by the backpropagation algorithm.

```
1. python serv.py # run on localhost:5000
2. Hold CTRL on keyboard and click on two circles to create connections
3. You can move circles on canvas
```

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/25-03-2025%20-%20EurekaLabs%20practice/91%20-%2025-03-2025%20-%20cd.png?raw=true)

Because it may be possible to perform and "tune" the trained network for more complex vocalizers, where there are 100k tokens and e.g. it has to predict tokens for this task like here where is the JAVASCRIPT code. Try copying part of the code from index.hml and pasting it here to see the tokens -> https://tiktokenizer.vercel.app/ . And when you move it on canvas, and this dynamically change weights / parameters and predictions then... 
