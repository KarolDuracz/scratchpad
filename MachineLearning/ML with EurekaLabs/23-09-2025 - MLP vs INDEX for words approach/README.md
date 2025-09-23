In theory, one neuron has inputs and outputs. And that's all it takes to make a decision. To calculate the next letter, word, token, etc. It's just math. Just for comparison with MLP. 
<br /><br />
What if we built an index of words and only calculated the prediction error for letters? For example, if the index under the letter "l" (lowercase l) contains possible combinations of "log","loss","let","length," then knowing the subsequent letters, we could calculate the error for the remainder and eliminate other possibilities (thus selecting the correct word) by writing subsequent letters.

