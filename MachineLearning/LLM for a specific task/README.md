Ok, ok. But how LLM can undestand TASK like this <br />
"Here you have a table with "tokens", cut out tokens 16, 12 and 11082 from this table and print the same thing only without these tokens." <br />
How GPT and LLM can undestand this command? And how to solve this task? <br />
Because this is similar to introduce of to C language. In C language you have "printf" command. And in C language you can create loop to iterate around this array and separate what you want to print an output. And in my opinion this is similar task, but from LLM perspective. BUT HOW IT FU... POSSIBLE that the model can understand the abstractions of such a command and actually write code similar to what a human could do after reading a book about C and a few hours of training - SOLVE THIS PARTICULAR PROBLEM ??? 
<br /><br />
Amazing.
<br /><br />
<i>But I have to admit that about 10 years ago I also tried to tame the "time series" to play the stock market hahaha. Before the transformer there was LSTM, GRU, also WaveNet (Andrej reproduced this net (https://www.youtube.com/watch?v=t3YJ5hKiMQ0), but also I read chapter in book about this architecture and WaveNet was good scores in music / sound with long period time / sequential . Better than LSTM and GRU. But I havent trained yet for this task WaveNet only for makemore exercies at this moment.). But at that moment I don't knew about that. I haven't thought about this problem any deeper at all. Ok, I knew there was decision tree, binary search, etc. I read about a model trained for Kinect to recognize gestures based on decision tree and I myself learned a small example algorithm to recognize several hand gestures (btw there is an article about it in PL magazine https://programistamag.pl/gestykinect/ ). But the Transformer era brings it all together. These techniques and algorithms. <b>But they used neural network (neurons) with parameters</b>. And magic happened. (...) And I was just thinking about how to pack OHLC and various other components of the "candle" on the chart into one value and  other things like that, like TOKENIZER does, but... this (transformer and GPT-2) solved a lot of stuff that for me was hard to solve 10 years ago. - BTW - I watched this recording (https://www.youtube.com/watch?v=N1TEjTeQeg0&ab_channel=UniversityofOxford). And I conclude that the models before AlexNet were based on SVM, decision trees etc. Mr. Hinton said that after the competition they called him to find out what model it was So... I haven't looked into it that much, but it was probably like that (https://en.wikipedia.org/wiki/AlexNet) . Another fact is Alex and Ilya they're was students of Mr. Hinton and Mr. Hinton himself had experimented in earlier years with the Markov chain, restricted Boltzmann machine. Papare about backpropagation appeared in the years
1986 (https://github.com/EurekaLabsAI/micrograd  |  https://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap8_PDP86.pdf) - I read about that in book about tensorflow and keras - ML TIMELINE  - (this note is for me - a small historical outline for me so to not to get lost ) ----> But when you deeper look into milestones one of the things that improve ML was (https://en.wikipedia.org/wiki/PyTorch) autograd engine which sped up what Andrej showed in this video Building makemore Part 4: Becoming a Backprop Ninja
 (https://www.youtube.com/watch?v=q8SA3rM6ckI). but what I found in internet, first tools came around 2012. This probably based on NumPy, Scipy, and other tools like that, But don't forget about keras and tensorflow which were before pytorch ! (...) But of course on the other side there is Yann Lecun and his model (https://github.com/karpathy/lecun1989-repro) . His Convolutional Network Demo from 1989
(https://www.youtube.com/watch?v=FwFduRA_L6Q&ab_channel=YannLeCun) and 2016 conference (https://www.youtube.com/watch?v=MkgPUWzpvi8&ab_channel=Talles) . Today this is "hello world" in machine learning, that is, the reproduction of this network for recognizing handwritten digits MNIST 0-9 (https://en.wikipedia.org/wiki/MNIST_database) .<br />
But there is on YT another video The Thinking Machine (Artificial Intelligence in the 1960s) https://www.youtube.com/watch?v=aygSMgK3BEM&ab_channel=RobertoPieraccini | The History of Artificial Intelligence [Documentary]
 https://www.youtube.com/watch?v=R3YFxF0n8n8&ab_channel=Futurology%E2%80%94AnOptimisticFuture
 <br /><br />
And of course the transformer era from 2017 but here is a small timeline and brief history development today's ML. <b>Today what I should be interested in is reading a ton of studies about models between 2010-2024. About transformer model etc. This should interest me today hah and then start it  train !!! </b> Ok, but seriously. This is a short outline for me . </i>
<br /><br />
$${\color{red}}$$	
		${{\color{red}\Huge{\textsf{     OK\   OK\   stop \  here\   and \   go\   to\   "library"\   ,I\    mean\   the\   internet\   hah  }}}}\$
<br />
last update : 17-09-2024
<br /><br />
update 24-09-2024 - I forgot about https://en.wikipedia.org/wiki/Bayesian_network | https://en.wikipedia.org/wiki/Markov_chain. This is not important right now, but in timeline of ML it is. It is more important to somehow make up for the shortcomings by the end of 2025. So all vision task, translator language task. Because task to create similar to like google.translate is good exercise.  But somehow this has to be sorted out by the end of 2025, as much as I can! There is more stuff here like this https://en.wikipedia.org/wiki/Probability_density_function | https://en.wikipedia.org/wiki/Spectral_density#Envelope | https://en.wikipedia.org/wiki/Linear_predictive_coding etc etc etc. There's a lot of interesting stuff here to learn next to ML. But...  (...) Also on wiki there is nice lecture and drawings about Perceptron and idea behind it https://en.wikipedia.org/wiki/Perceptron etc etc etc 
<br /><br />
<hr>
LINK TO LLM CODE SOURCE --> https://github.com/karpathy/llm.c/tree/master
<br /><br />

```
[3364]
[1235]
[353, 3028, 320, 66, 8, 220, 2366, 16, 12, 2366, 17, 578, 20774, 2298, 437, 5856, 4953, 627]
[353, 3028, 320, 66, 8, 220, 2366, 16, 12, 2366, 17, 45393, 13332, 198]
[353, 3028, 320, 66, 8, 220, 2366, 16, 12, 2366, 17, 79195, 38, 11, 4953, 627]
[1235]
[353, 10311, 1234, 279, 9091, 1914, 11, 6207, 220, 17, 13, 15, 320, 1820, 330, 10028, 803]
[353, 499, 1253, 539, 1005, 420, 1052, 3734, 304, 8907, 449, 279, 1914, 627]
[353, 1472, 1253, 6994, 264, 3048, 315, 279, 1914, 520, 198]
[1235]
[353, 257, 1795, 1129, 2185, 5206, 2726, 7116, 11082, 12, 17, 13, 15, 198]
[1235]
[353, 11115, 2631, 555, 8581, 2383, 477, 7378, 311, 304, 4477, 11, 3241, 198]
[353, 4332, 1234, 279, 1914, 374, 4332, 389, 459, 330, 1950, 3507, 1, 11643, 345]
[353, 6135, 7579, 2794, 11596, 3083, 4230, 9481, 11, 3060, 3237, 477, 6259, 627]
[353, 3580, 279, 1914, 369, 279, 3230, 4221, 10217, 8709, 323, 198]
[353, 9669, 1234, 279, 1914, 627]
[1235]
[353, 7030, 25, 4488, 13566, 366, 4075, 88, 31, 75, 359, 867, 916, 397]
[1235]
```

```
['/*\n',
 ' *\n',
 ' * Copyright (c) 2021-2022 The Khronos Group Inc.\n',
 ' * Copyright (c) 2021-2022 Valve Corporation\n',
 ' * Copyright (c) 2021-2022 LunarG, Inc.\n',
 ' *\n',
 ' * Licensed under the Apache License, Version 2.0 (the "License");\n',
 ' * you may not use this file except in compliance with the License.\n',
 ' * You may obtain a copy of the License at\n',
 ' *\n',
 ' *     http://www.apache.org/licenses/LICENSE-2.0\n',
 ' *\n',
 ' * Unless required by applicable law or agreed to in writing, software\n',
 ' * distributed under the License is distributed on an "AS IS" BASIS,\n',
 ' * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n',
 ' * See the License for the specific language governing permissions and\n',
 ' * limitations under the License.\n',
 ' *\n',
 ' * Author: Mark Young <marky@lunarg.com>\n',
 ' *\n']
```

```
[3364, 1235, 353, 3028, 320, 66, 8, 220, 2366, 16, 12, 2366, 17, 578, 20774, 2298, 437, 5856, 4953, 627, 353, 3028, 320, 66, 8, 220, 2366, 16, 12, 2366, 17, 45393, 13332, 198, 353, 3028, 320, 66, 8, 220, 2366, 16, 12, 2366, 17, 79195, 38, 11, 4953, 627, 1235, 353, 10311, 1234, 279, 9091, 1914, 11, 6207, 220, 17, 13, 15, 320, 1820, 330, 10028, 803, 353, 499, 1253, 539, 1005, 420, 1052, 3734, 304, 8907, 449, 279, 1914, 627, 353, 1472, 1253, 6994, 264, 3048, 315, 279, 1914, 520, 198, 1235, 353, 257, 1795, 1129, 2185, 5206, 2726, 7116, 11082, 12, 17, 13, 15, 198, 1235, 353, 11115, 2631, 555, 8581, 2383, 477, 7378, 311, 304, 4477, 11, 3241, 198, 353, 4332, 1234, 279, 1914, 374, 4332, 389, 459, 330, 1950, 3507, 1, 11643, 345, 353, 6135, 7579, 2794, 11596, 3083, 4230, 9481, 11, 3060, 3237, 477, 6259, 627, 353, 3580, 279, 1914, 369, 279, 3230, 4221, 10217, 8709, 323, 198, 353, 9669, 1234, 279, 1914, 627, 1235, 353, 7030, 25, 4488, 13566, 366, 4075, 88, 31, 75, 359, 867, 916, 397, 1235]
```


![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/LLM%20for%20a%20specific%20task/64%20-%2017-09-2024%20-%20trzeba%20tez%20rozkminic%20ten%20problem.png?raw=true)
