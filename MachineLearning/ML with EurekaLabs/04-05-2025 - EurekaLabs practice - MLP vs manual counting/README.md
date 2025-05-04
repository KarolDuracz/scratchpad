<h2>Hand counting combinations of e.g. 3 tokens in context vs. what MLP learns</h2>

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/04-05-2025%20-%20EurekaLabs%20practice%20-%20MLP%20vs%20manual%20counting/7%20-%2004-05-2025%20-%20cd.png?raw=true)

1. Link to chatgpt with prompts https://chatgpt.com/share/68174a5c-c3f0-8000-94d8-c6c5fa300237 
2. Link to notebook.ipynb https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/04-05-2025%20-%20EurekaLabs%20practice%20-%20MLP%20vs%20manual%20counting/Untitled8.ipynb
3. Dataset - https://github.com/EurekaLabsAI/mlp/blob/master/data/train.txt
4. MLP -> https://github.com/EurekaLabsAI/mlp/tree/master (python mlp.py)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/04-05-2025%20-%20EurekaLabs%20practice%20-%20MLP%20vs%20manual%20counting/04-05-2025%20-%20test%201.png?raw=true)

Back to https://pytorch.org/docs/stable/generated/torch.multinomial.html reason why etc
<hr>
We can change the context size at the every top in the CONTEXT LENGTH = 3 variable in .ipynb file. And then we see bellow first block sth like this when we start checking the range of 2-6 context sizes and what token appears after such a long sequence (But it still needs to be checked if it counts correctly): <br /><br />

```
// print(ngram_counter)
#2 - Total unique 2-grams: 623 - Counter({'n\n': 6322, 'a\n': 6253, 'an': 5067, '\na': 4135, 'e\n': 3702, 'ar': 3066, 'el': 3038,  ...
#3 - Total unique 3-grams: 6575 - Counter({'ah\n': 1622, 'na\n': 1568, 'on\n': 1401, 'an\n': 1394, '\nma': 1365, '\nja': 1186, '\nka': 1177, ...
#4 - Total unique 4-grams: 31305 - Counter({'yah\n': 461, 'lyn\n': 445, 'ana\n': 444, 'nna\n': 443, 'iah\n': 423, 'ynn\n': 406, 'anna': 401, 'lynn': 378, ...
#5 - Total unique 5-grams: 82032 - Counter({'lynn\n': 330, 'anna\n': 283, 'iyah\n': 266, 'leigh': 215, 'eigh\n': 201, 'iana\n': 183, 'elle\n': 182, 'ella\n': 173, ...
#6 - Total unique 6-grams: 142019 - Counter({'leigh\n': 187, 'ianna\n': 108, 'yanna\n': 74, 'elynn\n': 74, 'alynn\n': 65, 'liana\n': 60, '\nchris': 60, 'bella\n': 56, 'marie\n': 52, ...
```

Some time ago I checked what the plots look like for gradients for context 2-6 - https://github.com/EurekaLabsAI/mlp/issues/22
<br /><br />
And if we run the last block of code from the .ipynb file
then for context = 6 we see something like this. This is the probable behavior of the network in context = 6. But this is in theory. But if it works like that then some behaviors should scale and work the same on a larger model. The first one is leigh\n ! But we don't see here \n char but new line is execute. So this is next token for sequence leigh\n -> [a] . The letter 'a' has the most hits, 18. So "a, k, m, j, l" etc should appear most often.
```
Total unique 6-grams: 142019

Showing full 27-token distributions for the top 10 contexts:

Context: 'leigh
'  (count: 187)
    'a' : 18
    'k' : 17
    'm' : 16
    'j' : 14
    'l' : 14
    'e' : 12
    'r' : 11
    's' : 11
    'b' : 10
    't' : 8
    'c' : 7
    'g' : 7
    'h' : 7
    'p' : 6
    'z' : 6
    'i' : 5
    'd' : 4
    'f' : 4
    'n' : 4
    'q' : 2
    'v' : 2
    'o' : 1
    'y' : 1
    'u' : 0
    'w' : 0
    'x' : 0
   '\n' : 0

Context: 'ianna
'  (count: 108)
    'a' : 16
    'k' : 13
    'm' : 11
    'j' : 7
    't' : 7
    'c' : 5
    'g' : 5
    'h' : 5
    'n' : 5
    'p' : 5
    'b' : 4
    's' : 4
    'd' : 3
    'f' : 3
    'l' : 3
    'e' : 2
    'r' : 2
    'x' : 2
    'y' : 2
    'i' : 1
    'q' : 1
    'v' : 1
    'w' : 1
    'o' : 0
    'u' : 0
    'z' : 0
   '\n' : 0

Context: 'yanna
'  (count: 74)
    'a' : 9
    'l' : 7
    'm' : 7
    'k' : 6
    'j' : 5
    'b' : 4
    'd' : 4
    'o' : 4
    'e' : 3
    'r' : 3
    's' : 3
    't' : 3
    'z' : 3
    'c' : 2
    'h' : 2
    'n' : 2
    'v' : 2
    'w' : 2
    'g' : 1
    'q' : 1
    'y' : 1
    'f' : 0
    'i' : 0
    'p' : 0
    'u' : 0
    'x' : 0
   '\n' : 0

Context: 'elynn
'  (count: 74)
    'a' : 14
    'k' : 7
    'c' : 6
    'l' : 6
    'm' : 6
    'e' : 5
    'j' : 4
    'r' : 4
    'b' : 3
    'n' : 3
    'p' : 3
    'w' : 3
    'i' : 2
    's' : 2
    't' : 2
    'd' : 1
    'f' : 1
    'u' : 1
    'y' : 1
    'g' : 0
    'h' : 0
    'o' : 0
    'q' : 0
    'v' : 0
    'x' : 0
    'z' : 0
   '\n' : 0

Context: 'alynn
'  (count: 65)
    'k' : 8
    'a' : 6
    'b' : 5
    'l' : 5
    'm' : 5
    'r' : 5
    'e' : 4
    'z' : 4
    'c' : 3
    'd' : 3
    'h' : 3
    's' : 3
    't' : 3
    'j' : 2
    'x' : 2
    'i' : 1
    'n' : 1
    'o' : 1
    'p' : 1
    'f' : 0
    'g' : 0
    'q' : 0
    'u' : 0
    'v' : 0
    'w' : 0
    'y' : 0
   '\n' : 0

Context: 'liana
'  (count: 60)
    'j' : 12
    'a' : 11
    'k' : 5
    'm' : 5
    'c' : 4
    'b' : 3
    'd' : 3
    'r' : 3
    'e' : 2
    's' : 2
    'y' : 2
    'g' : 1
    'h' : 1
    'l' : 1
    'n' : 1
    't' : 1
    'w' : 1
    'x' : 1
    'z' : 1
    'f' : 0
    'i' : 0
    'o' : 0
    'p' : 0
    'q' : 0
    'u' : 0
    'v' : 0
   '\n' : 0

Context: '
chris'  (count: 60)
    't' : 45
    'h' : 4
    's' : 4
    'a' : 2
    'e' : 2
    'l' : 2
   '\n' : 1
    'b' : 0
    'c' : 0
    'd' : 0
    'f' : 0
    'g' : 0
    'i' : 0
    'j' : 0
    'k' : 0
    'm' : 0
    'n' : 0
    'o' : 0
    'p' : 0
    'q' : 0
    'r' : 0
    'u' : 0
    'v' : 0
    'w' : 0
    'x' : 0
    'y' : 0
    'z' : 0

Context: 'bella
'  (count: 56)
    'a' : 10
    'e' : 5
    'j' : 5
    'd' : 4
    'k' : 4
    'l' : 4
    'n' : 4
    't' : 3
    'z' : 3
    'c' : 2
    'm' : 2
    'r' : 2
    'b' : 1
    'g' : 1
    'i' : 1
    'o' : 1
    'p' : 1
    's' : 1
    'v' : 1
    'y' : 1
    'f' : 0
    'h' : 0
    'q' : 0
    'u' : 0
    'w' : 0
    'x' : 0
   '\n' : 0

Context: 'marie
'  (count: 52)
    'a' : 8
    'm' : 6
    'k' : 5
    's' : 5
    'r' : 4
    'j' : 3
    't' : 3
    'c' : 2
    'e' : 2
    'i' : 2
    'l' : 2
    'b' : 1
    'd' : 1
    'g' : 1
    'o' : 1
    'p' : 1
    'u' : 1
    'v' : 1
    'w' : 1
    'y' : 1
    'z' : 1
    'f' : 0
    'h' : 0
    'n' : 0
    'q' : 0
    'x' : 0
   '\n' : 0

Context: 'liyah
'  (count: 50)
    'a' : 8
    'k' : 7
    'm' : 4
    'n' : 4
    'b' : 3
    'd' : 3
    'e' : 3
    'j' : 3
    'c' : 2
    'r' : 2
    's' : 2
    't' : 2
    'z' : 2
    'f' : 1
    'h' : 1
    'l' : 1
    'p' : 1
    'y' : 1
    'g' : 0
    'i' : 0
    'o' : 0
    'q' : 0
    'u' : 0
    'v' : 0
    'w' : 0
    'x' : 0
   '\n' : 0
```

<hr>
<h2>TODO</h2>
That is, checking whether MLP actually learns these combinations and what appears, for example, after 3 tokens? In other words. If a neural network sees several such examples WHAT EXACTLY DOES IT DO? That is, how exactly does it behave (learn) when it sees similar "batches". etc.
<br /><br />
<i>btw. After changing my beliefs, and probably thinking correctly about NN, it's time to go deeper into these topics. The next plan will be topics like SGD vs AdamW. Why it can be trained on multi gpu systems, why SGD is not effective when distributing NN on several gpus. These are more complex issues of network training. But that's probably in the next months, when I get through the basics of what happens on small models like this one.</i>
