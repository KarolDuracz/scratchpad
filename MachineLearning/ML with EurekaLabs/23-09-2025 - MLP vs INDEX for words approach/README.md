In theory, one neuron has inputs and outputs. And that's all it takes to make a decision. To calculate the next letter, word, token, etc. It's just math. Just for comparison with MLP. 
<br /><br />
What if we built an index of words and only calculated the prediction error for letters? For example, if the index under the letter "l" (lowercase l) contains possible combinations of "log","loss","let","length," then knowing the subsequent letters, we could calculate the error for the remainder and eliminate other possibilities (thus selecting the correct word) by writing subsequent letters. How do you build such a word index? The typical approach is to take the first letter of the word. And based on subsequent letters, we reduce the number of possible combinations, which is clearly visible in the HTML demo.
<br /><br />
<b>The model training doesn't work optimally. This is just a demo for comparison.</b>

notebook with python code built on the pytorch Library - The first block, after training, simulates typing the words "log, loss" in few steps, i.e., char by char and probability distribution. The second listing, after training, allows us to type subsequent letters in the console and see possible word combinations.

https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/23-09-2025%20-%20MLP%20vs%20INDEX%20for%20words%20approach/MLP%20vs%20INDEX%20for%20words%20approach.ipynb

HTML interactive app ( no needs any serwer, it's just  runs on web browser ) - THE MODEL DOESN'T TRAIN WELL, BUT ON THIS WORD INDEX IT IS ENOUGH FOR IT TO PREDICT WORDS RATHER CORRECTLY. This is not a 100% working implementation like the one shown by Andrej in his Ngram or MLP training approach.

https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/23-09-2025%20-%20MLP%20vs%20INDEX%20for%20words%20approach/demo_app.html

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/23-09-2025%20-%20MLP%20vs%20INDEX%20for%20words%20approach/screen%20for%20demo%2023-09-2025.png?raw=true)

