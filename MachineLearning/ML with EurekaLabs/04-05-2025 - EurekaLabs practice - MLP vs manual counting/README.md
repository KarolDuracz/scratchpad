<h2>Hand counting combinations of e.g. 3 tokens in context vs. what MLP learns</h2>

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/04-05-2025%20-%20EurekaLabs%20practice%20-%20MLP%20vs%20manual%20counting/7%20-%2004-05-2025%20-%20cd.png?raw=true)

1. Link to chatgpt with prompts https://chatgpt.com/share/68174a5c-c3f0-8000-94d8-c6c5fa300237 
2. Link to notebook.ipynb https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/04-05-2025%20-%20EurekaLabs%20practice%20-%20MLP%20vs%20manual%20counting/Untitled8.ipynb
3. Dataset - https://github.com/EurekaLabsAI/mlp/blob/master/data/train.txt
4. MLP -> https://github.com/EurekaLabsAI/mlp/tree/master (python mlp.py)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/04-05-2025%20-%20EurekaLabs%20practice%20-%20MLP%20vs%20manual%20counting/04-05-2025%20-%20test%201.png?raw=true)

Back to https://pytorch.org/docs/stable/generated/torch.multinomial.html reason why etc

<h2>TODO</h2>
That is, checking whether MLP actually learns these combinations and what appears, for example, after 3 tokens? In other words. If a neural network sees several such examples WHAT EXACTLY DOES IT DO? That is, how exactly does it behave (learn) when it sees similar "batches". etc.
<br /><br />
<i>btw. After changing my beliefs, and probably thinking correctly about NN, it's time to go deeper into these topics. The next plan will be topics like SGD vs AdamW. Why it can be trained on multi gpu systems, why SGD is not effective when distributing NN on several gpus. These are more complex issues of network training. But that's probably in the next months, when I get through the basics of what happens on small models like this one.</i>
