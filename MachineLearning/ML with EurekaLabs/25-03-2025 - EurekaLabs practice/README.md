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
4. Press the left mouse button and move any circle to initialize the script.
```

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/25-03-2025%20-%20EurekaLabs%20practice/91%20-%2025-03-2025%20-%20cd.png?raw=true)

Because it may be possible to perform and "tune" the trained network for more complex dictionary, where there are 100k tokens and e.g. it has to predict tokens for this task like here where is the JAVASCRIPT code. Try copying part of the code from index.hml and pasting it here to see the tokens -> https://tiktokenizer.vercel.app/ . And when you move it on canvas, and this dynamically change weights / parameters and predictions then... 
<br /><br />
<i>But this is what needs to be done here, i.e. adding this dynamic change of parameters and predictions for subsequent letters, so that it generates subsequent "tokens" just like the trained model through backpropagation, when you move these circles or create connections. This is started in the serv.py part in the update_weights() function and sent a POST to the model. But the MLP model itself is only minimally here. NOT WORKING.
</i>
<br /><br />
So changin' that's on canvas must change DISTRIBUTION on softmax. This is the goal here.
<hr>
A few more words of explanation for this. In [The spelled-out intro to language modeling: building makemore] https://www.youtube.com/watch?v=PaCmpygFfXo&ab_channel=AndrejKarpathy Andrej started a makemore series. It ended with a backprop ninja. It was supposed to be GRU, LSTM, but eventually a manual backpropagation video was made which also seems like a better idea. But GRU and the basic Transformer are here --> https://github.com/karpathy/makemore/blob/master/makemore.py#L114
There is no Positional Encoding here from what I see, but https://www.tensorflow.org/text/tutorials/transformer?hl=en I think, after Attention layer this is most important part of the transformer that can imitate "reasoning" to predict next token based on context well. But that's not part of this demo. In this demo, only simple MLP can be tried to imitate instead of training with backpropagation.
