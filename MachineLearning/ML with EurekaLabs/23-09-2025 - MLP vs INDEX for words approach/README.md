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

Add more keywords, e.g. for JavaScript to HTML app ( line ~75 )

```
const WORDS = [
  // ECMAScript keywords & control
  "break","case","catch","class","const","continue","debugger","default","delete","do",
  "else","export","extends","finally","for","function","if","import","in","instanceof",
  "let","new","return","super","switch","this","throw","try","typeof","var","void",
  "while","with","yield","await","async","static","get","set","of",

  // common globals / constructors / types
  "console","math","number","string","boolean","array","object","json","date","promise",
  "map","set","weakmap","weakset","symbol","regexp","error","eval",

  // Promise / async helpers
  "then","catch","resolve","reject","async","await","finally",

  // Node / module-ish
  "module","exports","require","__dirname","__filename",

  // Common runtime & util functions
  "parseint","parsefloat","isnan","isfinite","encodeuri","decodeduri","encodeuricomponent",
  "decodeuricomponent","jsonstringify","jsonparse",

  // Math
  "floor","ceil","round","random","abs","min","max","pow","sqrt","log","exp",

  // common console helpers
  "log","warn","error","info","debug","table","time","timeend",

  // browser globals / DOM
  "window","document","location","history","navigator","localstorage","sessionstorage",
  "alert","confirm","prompt","fetch","addeventlistener","removeeventlistener",
  "queryselector","queryselectorall","getelementbyid","getelementsbyclassname","getelementsbytagname",
  "createelement","appendchild","removechild","replacechild","classlist","classname","dataset",
  "innerhtml","textcontent","value","style",

  // timers
  "settimeout","cleartimeout","setinterval","clearinterval",

  // Array methods
  "push","pop","shift","unshift","splice","slice","map","filter","reduce","foreach",
  "find","findindex","includes","indexof","join","split","concat","sort","reverse",

  // String methods
  "replace","tolowercase","touppercase","trim","substr","substring","startswith","endswith",
  "indexof","includes","split","concat",

  // common object / prototype helpers
  "prototype","constructor","hasownproperty","keys","values","entries","assign","create","defineproperty",

  // common DOM / UI helpers & patterns
  "addclass","removeclass","toggleclass","append","prepend","closest","matches","contains",

  // common libs / patterns (generic)
  "jquery","react","vue","angular","redux","rxjs",

  // misc identifiers often seen in JS code
  "id","name","type","length","size","index","key","value","payload","props","state","setstate",
  "dispatch","subscribe","unsubscribe","handler","callback","errorhandler","response","request"
].map(s => s.toLowerCase());
```

demo to show how many words there are under a given letter - https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/23-09-2025%20-%20MLP%20vs%20INDEX%20for%20words%20approach/word_index.html

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/23-09-2025%20-%20MLP%20vs%20INDEX%20for%20words%20approach/word%20index.png?raw=true)

Training the model for the larger "WORDS" array in the HTML demo takes a bit longer. For example, if the word "float" isn't present, the model tries to predict anything anyway.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/23-09-2025%20-%20MLP%20vs%20INDEX%20for%20words%20approach/extended%20WORDS%20table.png?raw=true)


