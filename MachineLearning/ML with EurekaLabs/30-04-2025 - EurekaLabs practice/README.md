<h2>to be continued</h2>
But technically it is a function that measures the error of calculations made by the network based on parameters. That's it
<br /><br />
This XOR problem etc here is about how to measure ERROR LOSS OF NETWORK PREDICTION > https://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap8_PDP86.pdf
<br /><br />
And this micrograd demo just shows WHAT A NETWORK DOES IF IT DOES NOT HAVE NON-LINEAR functions like Tanh, only linear layers https://github.com/EurekaLabsAI/micrograd . It's
simple but shows how NEURAL NETWORK works underhood with or without nonlinear functions. And this XOR problem is about this.
<br /><br />
But it's more complex because for example MNIST needs a nonlinear classifier. In turn, the transformer and text generation and using Positional Encoding,
Attention is a separate topic, which is a response to what recurrent networks did. It's damn complicated.
<br /><br />
Overall how a transformer, PE, Attention works is more complex than I thought... 
<br /><br />
That's all for now.
<br /><br />
Update 03-05-2025 - a few slides https://kcir.pwr.edu.pl/~witold/ai/mle_nndeep_s.pdf
<hr>
I think this is already correct thinking and I'm on the right path. But this is a damn deep topic to analyze. How they changed RNN to transformer and why they gave PE and Attention. I know why, but not much.
<br /><br />
After all this nonsense WHAT I WROTE HERE I ended up here, on <b>"Learning Internal Representations by Error Propagation"</b> and what these people improved in the Perceptrons: An Introduction to Computational Geometry by Marvin Minsky and Seymour Papert in 1969. And this XOR problem. I think I'm finally on the right path. (...) Plans, plans, but this is the CORE of it all, that these networks find a solution based on measuring the prediction error and improving the parameters so that the loss function is as small as possible. I tried to jump over everything but now I know why I got lost, since I did not understand what Rumelhart, Hinton and Williams were really doing
<br /><br />
OK, I think it's ok now and I think correctly. Now I need to educate myself.
<br /><br />
<i>plans plans plans, still plans... TODO.... but really it was enough to read those 45 pages from that PDF. I didn't plan that heh ;/ OK, now I know that I seriously need some time to delve into this because I really knew nothing.(...) With my current level of knowledge, it's better if I don't write anything more here, just educate myself better. Because it's irrelevant what I write now, since I don't know what the point of all this is. It doesn't matter if I figure out PE, attention. How does this solve the problem that RNN naturally had etc etc. It's not important now. It's doesn't matter at this moment of my current level of knowledge about ML. Some time ago I remembered some basic statistics, but I forgot. But this neural networks are something else... But it's good that I realized some things.</i>
<hr>
Update 01-05-2025 : Here I realized something while watching this recording from 10 years ago by Andrej https://github.com/KarolDuracz/scratchpad/tree/main/MachineLearning/ML%20with%20EurekaLabs/27-04-2025%20-%20EurekaLabs%20practice . Now I also know why I got so lost in the subject of ML. I thought wrong and there are still many things I probably don't know and I will learn in time when I start training networks, larger and larger models. I am also beginning to understand why AI, and how they are now trying to figure out LLM topics, is important. Because such an AUTONOMOUS system can be sent into space, instead of people. And such a perspective, e.g. AUTONOMOUS MARS ROVER SYSTEM and generally data analysis, introspection, communication, etc. etc. If this rover is left there alone, it must take actions autonomously. On the way there is also a flight between Earth <> Mars, this also requires planning of operations, calculations, etc. Only looking at it from the perspective of an AUTONOMOUS SYSTEM like this, can you realize something why AI field is going in this direction, creating these models like R1 with thinking process. There is a deeper vision and purpose behind this. But for now, these systems probably still need some fixes, they will still evolve towards this kind of autonomy. (...) But there are probably things along the way that need to be done around this meantime.
  <br /><br />
  Following this way of thought. Just look at how many instruments this rover has https://science.nasa.gov/mission/msl-curiosity/science-instruments/ and what operations must such a particular robot perform, including analysis of ground / soil composition, route planning, analysis of photos, surroundings, etc., etc. This is a spectrum of various skills that the model must have to be autonomous enough for this type of tasks. Maybe the entire LLM is an a seed of something that is heading towards AUTONOMY somewhere in the first ideas about AI... but definitely autonomous operation and analysis was probably the default idea and how the transformer learns patterns in the data... <b>BUT FOR ME THE MOST IMPORTANT NOW IS TO MOVE FROM THEORY TO PRACTICE AS QUICKLY AS POSSIBLE. AS SOON AS POSSIBLE. ENOUGH WITH THEORY AND PLANNING. IT'S TIME TO MOVE TO LEARNING AND ACTION. FOR ME THIS IS IMPORTANT RIGHT KNOW. GET BASICS. IMPLEMENTATIONS. CODE. MATHS. </b>
<br /><br />
And it's good that I finally understood that I need to take on the basics --> <b>"Learning Internal Representations by Error Propagation"</b> and to understand how networks learn to solve problems like this example by Andrej from micrograd who classifies 3 colors on canvas. There is still a lot of fun to be had with this, as you can make the network larger, add more colors, test linear approaches, different activation functions, etc., etc. To learn something from it.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/30-04-2025%20-%20EurekaLabs%20practice/micrograd%20js%20demo%20classifier.png?raw=true)

But it's good that I finally started thinking about it properly, thanks to Andrej and his work. Time to delve into it seriously :)
