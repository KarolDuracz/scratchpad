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

 Total unique n-grams, means, that there are actually so many unique combinations. So for 2-grams there are 623 combinations? For 3-gram already 6475? So most of it is repeated for a small context, and for larger the context for this database ... what? (In https://github.com/EurekaLabsAI/ngram we see : "Our dataset is that of 32,032 names from ssa.gov for the year 2018, which were split into 1,000 names in the test split, 1,000 in val split, and the rest in the training split, all of them inside the data/ folder." - Maybe that's why context #4 has the least loss in the link below 2.00298 for AdamW + Tanh. Because if there are 31k names, that's the number of combinations.... But I don't know if it matters)
<br /><br />
<i>I moved the image from this place to the bottom of the page because it is disruptive [9 - 05-05-2025 - another pics.png]</i>
<br /><br />
Some time ago I checked what the plots look like for gradients for context 2-6 - https://github.com/EurekaLabsAI/mlp/issues/22
<br /><br />
[ <i>From this point on I am writing about what is below, from this point on, don't look up, only down the page</i> ] And if we run the last block of code from the .ipynb file
then for context = 6 we see something like this. This is the probable behavior of the network in context = 6. But this is in theory. But if it works like that then some behaviors should scale and work the same on a larger model. The first one is leigh\n ! But we don't see here \n char but new line is execute. So this is next token for sequence leigh\n -> [a] . The letter 'a' has the most hits, 18. So "a, k, m, j, l" etc should appear most often. <b>If I think correctly, MLP using backprop should learn these behaviors for any sequence length (context length), as in what we see "counting by hand", or something close to that in prediction</b>. But this is training data. What you see here in context 6 or 4 https://github.com/EurekaLabsAI/mlp/issues/22 are the predictions of the model after TRAINING ON THIS DATA. So looking at what is below for context 6 and 4 we really see what we should expect from these sequences, what it should learn for a given sequence length. I guess.
<br /><br />
Here is manual counting for 6 nad 4 context length. In this https://raw.githubusercontent.com/EurekaLabsAI/mlp/refs/heads/master/data/train.txt file. And for context 4, below we see the sequence "ANNA" (without the \n sign, just anna). And when we open it in the browser as RAW and press CTRL + F to open inline ejection and search for "anna", there are indeed 401 finds. And these are the most common 3 repeating tokens after this sequence '\n' : 283, 'h' : 46, 'l' : 35. So if the model generates anna (or we give it in prompt) it should do one of these 3 using ~multinomial distribution. I guess. (...) '\nmar'  (count: 374) - This is also an interesting case, because there are probably 374 sequences of this type where \n + mar is first in train.txt. But overall, there are already 735 combinations of mar (3 letters, not 4). So almost half of them have the \n character before this combination. So? What does the network learn from this?

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

CONTEXT = 4 (counting by hand) 

```
Total unique 4-grams: 31305

Showing full 27-token distributions for the top 10 contexts:

Context: 'yah
'  (count: 461)
    'a' : 76
    'k' : 44
    'm' : 39
    'j' : 37
    'b' : 24
    'e' : 24
    'r' : 23
    'l' : 22
    's' : 21
    'd' : 17
    'z' : 17
    'n' : 16
    'c' : 15
    't' : 13
    'g' : 10
    'h' : 10
    'p' : 10
    'y' : 10
    'f' : 9
    'v' : 9
    'i' : 5
    'o' : 4
    'w' : 3
    'u' : 2
    'q' : 1
    'x' : 0
   '\n' : 0

Context: 'lyn
'  (count: 445)
    'a' : 65
    'k' : 47
    'm' : 35
    'c' : 30
    'e' : 26
    'd' : 24
    's' : 23
    'l' : 22
    'r' : 21
    'j' : 19
    't' : 19
    'b' : 16
    'z' : 14
    'g' : 12
    'n' : 12
    'h' : 11
    'p' : 10
    'v' : 10
    'i' : 9
    'o' : 5
    'f' : 4
    'y' : 4
    'q' : 3
    'x' : 2
    'u' : 1
    'w' : 1
   '\n' : 0

Context: 'ana
'  (count: 444)
    'a' : 59
    'j' : 42
    'm' : 36
    'k' : 35
    's' : 33
    'c' : 26
    'r' : 26
    'd' : 23
    'l' : 23
    't' : 19
    'z' : 19
    'e' : 17
    'b' : 14
    'n' : 14
    'g' : 10
    'h' : 10
    'i' : 6
    'p' : 6
    'y' : 6
    'f' : 5
    'w' : 5
    'o' : 4
    'x' : 4
    'v' : 2
    'q' : 0
    'u' : 0
   '\n' : 0

Context: 'nna
'  (count: 443)
    'a' : 54
    'k' : 46
    'm' : 40
    'b' : 26
    's' : 26
    'd' : 23
    'j' : 23
    'l' : 23
    'r' : 21
    't' : 21
    'e' : 20
    'c' : 19
    'h' : 16
    'n' : 16
    'z' : 14
    'f' : 9
    'p' : 9
    'g' : 8
    'w' : 6
    'y' : 6
    'o' : 5
    'v' : 5
    'i' : 3
    'q' : 2
    'x' : 2
    'u' : 0
   '\n' : 0

Context: 'iah
'  (count: 423)
    'a' : 59
    'j' : 35
    'e' : 32
    'k' : 31
    'm' : 28
    'r' : 25
    'c' : 24
    's' : 24
    'd' : 23
    't' : 22
    'l' : 18
    'z' : 16
    'b' : 15
    'n' : 15
    'g' : 11
    'h' : 9
    'y' : 6
    'f' : 5
    'p' : 4
    'q' : 4
    'v' : 4
    'w' : 4
    'i' : 3
    'o' : 3
    'u' : 2
    'x' : 1
   '\n' : 0

Context: 'ynn
'  (count: 406)
    'a' : 56
    'k' : 37
    'm' : 31
    'j' : 25
    'r' : 24
    'c' : 23
    'l' : 22
    't' : 21
    'e' : 20
    's' : 20
    'h' : 14
    'b' : 13
    'd' : 13
    'n' : 12
    'g' : 11
    'p' : 11
    'i' : 8
    'z' : 8
    'f' : 7
    'o' : 7
    'y' : 7
    'x' : 5
    'v' : 4
    'w' : 4
    'u' : 2
    'q' : 1
   '\n' : 0

Context: 'anna'  (count: 401)
   '\n' : 283
    'h' : 46
    'l' : 35
    'b' : 5
    'n' : 5
    's' : 5
    'm' : 4
    't' : 3
    'y' : 3
    'e' : 2
    'k' : 2
    'r' : 2
    'c' : 1
    'd' : 1
    'g' : 1
    'i' : 1
    'j' : 1
    'v' : 1
    'a' : 0
    'f' : 0
    'o' : 0
    'p' : 0
    'q' : 0
    'u' : 0
    'w' : 0
    'x' : 0
    'z' : 0

Context: 'lynn'  (count: 378)
   '\n' : 330
    'e' : 37
    'a' : 3
    'l' : 3
    'o' : 3
    'i' : 1
    's' : 1
    'b' : 0
    'c' : 0
    'd' : 0
    'f' : 0
    'g' : 0
    'h' : 0
    'j' : 0
    'k' : 0
    'm' : 0
    'n' : 0
    'p' : 0
    'q' : 0
    'r' : 0
    't' : 0
    'u' : 0
    'v' : 0
    'w' : 0
    'x' : 0
    'y' : 0
    'z' : 0

Context: '
mar'  (count: 374)
    'i' : 104
    'y' : 46
    'l' : 44
    'c' : 34
    'k' : 27
    't' : 18
    'g' : 15
    'q' : 14
    'a' : 13
    's' : 13
    'e' : 12
    'v' : 11
    'j' : 5
    'r' : 4
    'w' : 4
    'n' : 2
    'u' : 2
    'z' : 2
    'b' : 1
    'd' : 1
    'x' : 1
   '\n' : 1
    'f' : 0
    'h' : 0
    'm' : 0
    'o' : 0
    'p' : 0

Context: 'ani
'  (count: 348)
    'a' : 45
    'm' : 32
    's' : 27
    'k' : 24
    'd' : 23
    'j' : 23
    'c' : 18
    'e' : 16
    't' : 16
    'b' : 15
    'r' : 15
    'l' : 14
    'n' : 12
    'g' : 10
    'h' : 10
    'z' : 9
    'i' : 8
    'p' : 8
    'y' : 6
    'o' : 5
    'f' : 3
    'v' : 3
    'u' : 2
    'w' : 2
    'q' : 1
    'x' : 1
   '\n' : 0
```

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/04-05-2025%20-%20EurekaLabs%20practice%20-%20MLP%20vs%20manual%20counting/9%20-%2005-05-2025%20-%20another%20pics.png?raw=true)

<hr>
To have a quick comparison and not do the same thing twice again and again. 
<br /><br />
Default model from repo, I didn't change anything in network settings, 50000 iterations of learning loop, only line 193 - context_length = 4 and 6 and 8 - instead of 3 -> https://github.com/EurekaLabsAI/mlp/blob/master/mlp_pytorch.py#L193C1-L193C19

Because that's what I checked here, i.e. 4 and 6. Context 8 additionally by the way.

and 10000 characters are generated instead of 200, in this loop, line 244 - https://github.com/EurekaLabsAI/mlp/blob/master/mlp_pytorch.py#L244

I am most interested in the prediction for context 4 and 6 and the occurrence of TOP 10 sequences from the examples above, e.g. "leigh\n" for context_length=6 and "yah\n" for context_length=4

CONTEXT LENGTH 4 -> https://raw.githubusercontent.com/KarolDuracz/scratchpad/refs/heads/main/MachineLearning/ML%20with%20EurekaLabs/04-05-2025%20-%20EurekaLabs%20practice%20-%20MLP%20vs%20manual%20counting/predictions/context_length-4.txt <br />
CONTEXT LENGTH 6 -> https://raw.githubusercontent.com/KarolDuracz/scratchpad/refs/heads/main/MachineLearning/ML%20with%20EurekaLabs/04-05-2025%20-%20EurekaLabs%20practice%20-%20MLP%20vs%20manual%20counting/predictions/context_length-6.txt <br />
CONTEXT LENGTH 8 -> https://raw.githubusercontent.com/KarolDuracz/scratchpad/refs/heads/main/MachineLearning/ML%20with%20EurekaLabs/04-05-2025%20-%20EurekaLabs%20practice%20-%20MLP%20vs%20manual%20counting/predictions/context_length-8.txt 
<br /><br />
We can quickly check what I wrote above, e.g. "mar" etc.
<hr>
<h2>Summary for this demo</h2>
People behind the first models (https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf - 2003) didn't have the tools that we have today. And the computers we have today. I didn't go into details about the hardware, but my laptop from 2011 probably has several times greater performance than those from over 20 years ago. Besides, training Lenet on MNIST takes a few moments. And CHATGPT. But since it's here ->
https://chatgpt.com/share/6818bfa1-aae8-8000-9250-3a9496071230
<br /><br />
First I asked questions about the model itself and the number of parameters. And as gpt writes. Total = 1296 + 73728 + 512 + 13824 + 27 = 89387 parameters. 

```
 == total parameters ==
wte ==> (27, 48) | Parameters: 1296
fc1_weights ==> (144, 512) | Parameters: 73728
fc1_bias ==> (512,) | Parameters: 512
fc2_weights ==> (512, 27) | Parameters: 13824
fc2_bias ==> (27,) | Parameters: 27
Total Parameters: 89387
```

Then I gave him these observations about the number of combinations for context 2-6 on this data. And he gave answers to these questions. Interestingly, according to him for 27 ** 6 = 387,000,000 combinations where in this data I have an upper limit of ~22,000 looking at this image above where after 10 tokens in the context you can already see a "flat line". But it will be time to better understand what is behind it and finally read "A Neural Probabilistic Language Model" from 2003. Because understanding this is a step to bigger models and what happened next, where we are today.
<br /><br />
There are probably some more mathematical and data-related nuances, so I'll try to "torture" this MLP further. But + for me, that from theory and nonsense finally "a bit more technical" post. So overall there is a little progress.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/04-05-2025%20-%20EurekaLabs%20practice%20-%20MLP%20vs%20manual%20counting/summary%20picture.png?raw=true)

<hr>
Ok, so I'm counting manually vs. MLP. The image above shows that it doesn't pay off above 10 combinations, i.e. context 10 max. But what does the prediction look like for context_length = 50 and what does it actually predict after training MLP? Since I did here this iterration for 50, so why not as summary check also this.
<br /><br />
But unfortunately ERROR. I guess there is "hard coding" for no more than 8 context size [ ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 2400 is different from 384) ] - that means for 384/48 = 8 but for 2400/48 = 50. That means, it tries to count for context 8 but I give it 50. Here is OUTPUT log -> https://raw.githubusercontent.com/KarolDuracz/scratchpad/refs/heads/main/MachineLearning/ML%20with%20EurekaLabs/04-05-2025%20-%20EurekaLabs%20practice%20-%20MLP%20vs%20manual%20counting/predictions/context_length%2050%20-%20error.txt

<hr>
<h2>TODO</h2>
That is, checking whether MLP actually learns these combinations and what appears, for example, after 3 tokens? In other words. If a neural network sees several such examples WHAT EXACTLY DOES IT DO? That is, how exactly does it behave (learn) when it sees similar "batches". etc.
<br /><br />
<i>btw. After changing my beliefs, and probably thinking correctly about NN, it's time to go deeper into these topics. The next plan will be topics like SGD vs AdamW. Why it can be trained on multi gpu systems, why SGD is not effective when distributing NN on several gpus. These are more complex issues of network training. But that's probably in the next months, when I get through the basics of what happens on small models like this one. (...) btw. Why do I want to do any more SGD vs. AdamW etc tests? Because that's the engine that runs underneath these networks. And I don't really know anything, except that it's used in backpropagation. And there are some nuances associated with this. It's not about grinding this topic, but SGD was previously used for training. Now AdamW. from what I see in the 2017 study "Attention Is All You Need" they also used AdamW https://arxiv.org/pdf/1706.03762. Andrej's toy demos use the AdamW implementation instead of SGD. And I'll probably rework this topic [Building makemore Part 4: Becoming a Backprop Ninja] https://www.youtube.com/watch?v=q8SA3rM6ckI&ab_channel=AndrejKarpathy - as an exercise because in the implementation with CPU <> GPU it will be needed, when you do something "from scratch"... This is the main goal for these exercises SGD vs AdamW in next months.</i>
<br /><br />
Since "ясность есть" and it is more or less known why there is SGD <> AdamW for the network, if we look at it from the perspective of what should be when calculating manually, e.g. for context = 3 vs what such a small MLP learns. And how it does it. The next step is a MORE COMPLEX DATA SET (text). Andrej gave "tiny shakespeare" https://github.com/karpathy/nanoGPT/tree/master/data/shakespeare as a training set https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt. Just like before I did not understand how important it is to sign images by the network, e.g. "bird" <> "a bird sitting on a branch". That is what I wrote in the previous post about the NVIDIA presentation from 10 years ago. Similarly 2 years ago when the nanoGPT repo first appeared I did not understand why "Shakespeare" ?! And now I know... that it is a more complex text, because there are characters (persons) in it, there is some context for each character, different plots, AND SIMPLY PREDICTING A SEQUENCE from ngrams as above, it will not learn TEXT UNDERSTANDING. It will only predict subsequent characters (char by char) of the text without INSIGHTING INTO THE MEANING of what is happening in it.
<br /><br />
[ It has less significance now. But thanks to this exercise "MLP vs manual counting" I also understood that ] Comparing this MLP model (trained on 32,032 names from ssa.gov) vs nanoGPT (tiny shakespeare)... I used to not understand how a trained model can then finish a sequence or how GPT-2 responds to "prompts". But the same behavior can be observed on this small MLP. Only on this small MLP will we generate a prediction e.g. for ngram=4 i.e. [ 1, 18, 18, 1 ] -> [ 27] i.e. 'anna' -> '\n'. For this sequence the model will "remember" the combination in the parameters. And it works the same way on a larger network, only the difference is in the architecture and DATA ON WHICH THIS MODEL IS TRAINED. So if I now train a larger model like nanoGPT on "tiny shakespear" data and feed the starting prompt with random words/tokens/sequences taken out of context, the model will continue to generate predictions... AND HERE BEGINS THE QUESTION ABOUT TRANSFORMER ARCHITECTURE etc.
<br /><br />
to be continued...
 <br /><br />

$${\color{red}}$$	
		${{\color{red}\Huge{\textsf{    [CLOSED]}}}}\$
