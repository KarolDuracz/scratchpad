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
[  <i>It has less significance now. But thanks to this exercise "MLP vs manual counting" I also understood that: </i>] Comparing this MLP model (trained on 32,032 names from ssa.gov) vs nanoGPT (tiny shakespeare)... I used to not understand how a trained model can then finish a sequence or how GPT-2 responds to "prompts". But the same behavior can be observed on this small MLP. Only on this small MLP will we generate a prediction e.g. for ngram=4 i.e. [ 1, 18, 18, 1 ] -> [ 27] i.e. 'anna' -> '\n'. For this sequence model will "remember" the combination, in the parameters. And it works the same way on a larger network, only the difference is in the architecture and DATA ON WHICH THIS MODEL IS TRAINED. So if I now train a larger model like nanoGPT on "tiny shakespear" data and feed the starting prompt with random words/tokens/sequences taken out of context, for example "MENENIUS: I tell you, friend", I should observe similar behavior, model will continue to generate predictions... AND HERE BEGINS THE QUESTION ABOUT TRANSFORMER ARCHITECTURE, AND HOW THE NETWORK INTERNALLY LEARNS, i.e. SGD, AdamW, and why or not it learns etc. etc.
<br /><br />
to be continued...
 <br /><br />

$${\color{red}}$$	
		${{\color{red}\Huge{\textsf{    [CLOSED]}}}}\$

 <br /><br />
Update 06-05-2025 - btw. This is another thing that I saw somewhere along the way in Andrej's work "surfing the internet" -  http://karpathy.github.io/2015/05/21/rnn-effectiveness/ - Now it will be interesting to read from the perspective of current knowledge how small MLP "behaves" vs what Andrej presents in this post and especially those colorful pictures at the bottom e.g. showing the Linux code.
<br /><br />
And one more thing. Andrej demonstrated exactly this in the video https://youtu.be/PaCmpygFfXo?t=6517 [The spelled-out intro to language modeling: building makemore] around 1:48:39 - but he showed it on "bigram model" vs MLP. This is the same analogy. But here you can also "manually" use other combinations, not only for 2 characters https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/04-05-2025%20-%20EurekaLabs%20practice%20-%20MLP%20vs%20manual%20counting/Untitled8.ipynb
<br /><br />
Update 08-05-2025 - Andrej in the MLP repo on eurekalabs gave 3 exercises, making a graphs, implementing it in C and finding better parameters. I wanted to approach making CUDA code on Google Colab but first I decided to try it on my own hardware with code, e.g. Linux (https://github.com/torvalds/linux). Initially I made a script to "walk through" all folders, find "*.c" files, open them, copy everything to one file and so on through the entire Linux repo. And I stopped at ~700 MB txt file. Then another script which cut it into pieces of 50 MB and when I opened the first one, after running this file which counts "manually" the number of Untitled8.ipynb combinations I got these results : 

```
Total unique 1-grams: 128 | 700 MB RAM
Total unique 3-grams: 145131 | 3.5 GB RAM 
Total unique 4-grams: 802295 | x
Total unique 10-grams: 13888880 | 4.8 GB RAM 
Total unique 15-grams: 24777100 | 5.5 GB RAM
```

So compared to the name database, there are no longer only 27 combinations for CONTEXT = 1, only 128 because there are more characters in the text than just a-z. And when I started trying other numbers of combinations, at 15 my computer is already at the RAM limit. But that was the straightforward attempt (naive). Looking at this post above, trying to set better parameters for the name database probably won't tell me much. I think I need to experiment with a different type of data (text) to learn more. But this is still the ngram approach, without learning to understand text. 
<br /><br />
This is what the first 2000 characters from file 1/14 look like

```
>>> f = open("part_0.txt", "rb").read()
>>> f[:2000]
b'/* ===== File 1: C:\\Users\\kdhome\\Documents\\progs\\EurekaLabs\\08-05-2025 - analyze linux repo\
\linux-master\\linux-master\\arch\\alpha\\boot\\bootp.c ===== */\r\n// SPDX-License-Identifier: GPL-
2.0\r\n/*\r\n * arch/alpha/boot/bootp.c\r\n *\r\n * Copyright (C) 1997 Jay Estabrook\r\n *\r\n * Thi
s file is used for creating a bootp file for the Linux/AXP kernel\r\n *\r\n * based significantly on
 the arch/alpha/boot/main.c of Linus Torvalds\r\n */\r\n#include <linux/kernel.h>\r\n#include <linux
/slab.h>\r\n#include <linux/string.h>\r\n#include <generated/utsrelease.h>\r\n#include <linux/mm.h>\
r\n\r\n#include <asm/console.h>\r\n#include <asm/hwrpb.h>\r\n#include <asm/io.h>\r\n\r\n#include <li
nux/stdarg.h>\r\n\r\n#include "ksize.h"\r\n\r\nextern unsigned long switch_to_osf_pal(unsigned long
nr,\r\n\tstruct pcb_struct *pcb_va, struct pcb_struct *pcb_pa,\r\n\tunsigned long *vptb);\r\n\r\next
ern void move_stack(unsigned long new_stack);\r\n\r\nstruct hwrpb_struct *hwrpb = INIT_HWRPB;\r\nsta
tic struct pcb_struct pcb_va[1];\r\n\r\n/*\r\n * Find a physical address of a virtual object..\r\n *
\r\n * This is easy using the virtual page table address.\r\n */\r\n\r\nstatic inline void *\r\nfind
_pa(unsigned long *vptb, void *ptr)\r\n{\r\n\tunsigned long address = (unsigned long) ptr;\r\n\tunsi
gned long result;\r\n\r\n\tresult = vptb[address >> 13];\r\n\tresult >>= 32;\r\n\tresult <<= 13;\r\n
\tresult |= address & 0x1fff;\r\n\treturn (void *) result;\r\n}\t\r\n\r\n/*\r\n * This function move
s into OSF/1 pal-code, and has a temporary\r\n * PCB for that. The kernel proper should replace this
 PCB with\r\n * the real one as soon as possible.\r\n *\r\n * The page table muckery in here depends
 on the fact that the boot\r\n * code has the L1 page table identity-map itself in the second PTE\r\
n * in the L1 page table. Thus the L1-page is virtually addressable\r\n * itself (through three leve
ls) at virtual address 0x200802000.\r\n */\r\n\r\n#define VPTB\t((unsigned long *) 0x200000000)\r\n#
define L1\t((unsigned long *) 0x200802000)\r\n\r\nvoid\r\npal_init(void)\r\n{\r\n\tunsigned long i,
rev;\r\n\tstruct percpu_struct * percpu;\r\n\tstruct pcb_struct * pcb_pa;'
>>>
```

For comparison with the name database. (...) But it is important to know here that FIXED WINDOW is used, so it just loops through the character sequences in this demo, with a window size of e.g. 4 [ x, x, x, x ] -> [ x ] to predict the next 1 . And that these batches are "randomly" taken from the text. If there is such an approach. And the database is "shuffled". But I didn't do that here. Keep it mind. This is just to have a comparison.
<br /><br />
So for context length 15 it looks like this
```
       15              1
[address & 0x1ff] --> [f]
```

So it's not as "smart" as the transformer with BPE tiktoken . But before this architecture was created...
<br /><br />
Summary for linux repo - https://github.com/KarolDuracz/scratchpad/tree/main/MachineLearning/ML%20with%20EurekaLabs/04-05-2025%20-%20EurekaLabs%20practice%20-%20MLP%20vs%20manual%20counting/linux%20repo - to get TOP 10 combinations for context 3,4,5 to see what is e.g. Context: 'truct' (count: 107070) 'i' : 5061 'u' : 1319 '\n' : 125 etc. in this naive approach. (...) For ngram=5 this looks interesting : 

```
Context: 'struc'  (count: 107039) -> (what letter do you expect?)
Context: 'retur'  (count: 86937)  -> (what letter do you expect?)
```

OK, that's it for this exercise. (...) But that's my way of learning to understand it. After all, now I need to delve into the mathematical aspects. Without that, I can't further understand how it works. But 1 step forward.
<br /><br />
Update 10-05-2025 - I need to be sure that I'm thinking correctly and that what I'm writing actually works as I described. I'm not 100% sure about this yet, so I checked HOW MANY TIMES IN THE TRAINING LOOP the model sees (learns) these [ x, x, x ] with a window for context, let's say 3. Because that was the first thing I measured in the images at the very top. I first measured CONTEXT=3. And how does it look for combinations from the TOP 30, from first place vs. place 30. So the combination 'ah\n' vs. '\nbr.
<br /><br />
[ <i>look at the picture "test a" in this folder ah\n vs \nbr</i> ] I only measured the first 400 steps on the left window and at what indexes and on which step it sees. For 'ah\n' after 400 steps it comes out 379 times that it sees the sequence [ 1, 8, 0 ] i.e. [ 'a', 'h', '\n' ]. So on the right window I did a second test, a bit longer to see how much it comes out after, say, 3400 steps, it turned out that the model sees this combination 3310 times.
<br /><br />
[ <i>look at the picture "test b"</i> ]The same for the combination '\nbr', i.e. [ 0, 2, 18 ]. The left window shows how it looked at this combination in the BATCH_SIZE loop, which has (128, 3), size. That is, after 400 steps it saw this combination 134 times. After 3400 steps it saw this combination 1233 times.
<br /><br />
This means that it actually has an effect. But to be 100% sure about this combination, you still need to check after 50,000 steps what the difference is. Images and code are here : 
https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/04-05-2025%20-%20EurekaLabs%20practice%20-%20MLP%20vs%20manual%20counting/ah%5Cn%20vs%20%5Cnbr/README.md

```
AFTER 50.000 steps with BATCHSIZE = 128 and CONTEXT_LENGTH = 3
'ah\n' - [ 1, 8, 0 ] - p: 48554   n: 6351318
// (48554 + 6351318) / 128 = 49999.0 - ok
// (positive + negative) / batch_size == 50000 steps
// (48554 / ((48554 + 6351318)) = 0.00758 - less than 1% ( 0.0075 * 100 = 0.75 )

'\nbr' - [ 0, 2, 18 ] - p: 18114   n: 6381758
// (18114 + 6381758) / 128 = 49999.0 - ok
// (18114 / (18114 + 6381758)) = 0.00283 - less tha 1% ( 0.002 * 100 = 0.2)

48554 / 18114 = 2.68 - 2.68 times more "events" in the training loop for 'ah\n' than '\nbr'
----------------------------------------
Looking at what is below the graph in the Untitled8.ipynb file for TOP 30 and CONTEXT_LENGTH = 3

Counter({'ah\n': 1622, 'na\n': 1568, 'on\n': 1401, 'an\n': 1394, '\nma': 1365, 
'\nja': 1186, '\nka': 1177, 'en\n': 1149, 'lyn': 910, 'yn\n': 904, 'ari': 895, 
'a\na': 870, 'n\na': 860, 'ia\n': 859, 'ie\n': 796, 'ell': 767, 'ann': 765, 'ana': 754, 
'ian': 740, 'mar': 735, 'in\n': 717, 'el\n': 688, 'ya\n': 675, 'ani': 666, 'la\n': 649, 
'\nda': 648, 'iya': 636, 'er\n': 634, 'n\nk': 609, '\nbr': 605,

we see exactly these combinations and results FOR MANUAL COUNTING:
'\nbr': 605
'ah\n': 1622

So, 1622/605 = 2.68 <--- THIS IS EXACTLY THE SAME NUMBER which comes out after 50,000 iterations with batchsize 128
----------------------------------------
I don't know where these values ​​come from yet, but after 50,000 steps vs. manual counting it turns out that there
are ~29 times more "events" in the training loop for this 2 examples
48554 / 1622 = 29.934
18114 / 605 = 29.940 
```
So with BATCH_SIZE = 128 the model sees this batch even several times in 1 step, as is the case with 'ah\n'. 
That's why I woke up about this PIPELINE in the previous post. Because I just realized that this algorithm has been improved by adding these MINI BATCH etc.
That's one of the reasons I wrote in previous posts that I thought in wrong way.
OK, that's it. Because too much information at once is also not good.
<br /><br />
I think that this is enough information to deal with the network training algorithm and the network itself in the next exercise. And the mathematics behind it.

<h3>Last words as summary - 10-05-2025 - </h3>
I'm looking at this video again now [ The spelled-out intro to language modeling: building makemore ] and in minute ~12:23 https://youtu.be/PaCmpygFfXo?t=742 Andrej did the same statistic for the bigram model. Only I approached it again, only I started with not context_length=2, but 3. n\n ('n', '<E>') 6763, a\n ('a', '<E>') 6648 and so on that we see on this video. But that was before split dataset into train/eval/test. Similar to what we see at the top of the page (Counter({'n\n': 6322, 'a\n': 6253, ...). But along the way I started to wonder "what does it really mean and how does the model learn these mini batches", these "context windows" with e.g. 3 ngram.
