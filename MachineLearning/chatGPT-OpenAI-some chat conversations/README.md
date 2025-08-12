12-08-2025 - GPT-5 - Still hallucinate. It was a temporary chat, so I don't have the link, but GPT-5 ( free version with reasoning ) made a few mistakes when generating answers, but when it came to generating the code for testing, it finally provided the correct answer. This code works and provides accurate results. The question was posed in a way that made it seem like the user didn't fully understand how the first layers, for example, are calculated with 50,257 tokens. And the model correctly pointed user in the right direction: each token isn't a single value, i.e., a token ID in the vocab - Each token is represented by a vector of parameters ( <b> 768 </b> x 50,257 = 38,597,376 ) . I asked about the differences between the tokenizer from GPT-2 to GPT-5. He started answering that the first one had 50,257, then 100,277 in GPT3, ~199.997-200,000 in GPT4, and so on. He tried to approximate what's in GPT5 because he couldn't find information online. He gave 300k :) ( <b>at this point, hallucinating is useful if needs to approximate </b>). But then he gave incorrect values several times, including the differences in converting words to integers in the tokenizer. "He" was off by an average of 50, instead of 152, for r50k_base, model gave sth about 102. Then model misinterpreted how he "cuts" the tokens (as seen in the image). But at the same time, model gave the correct code. ---- What is interesting. If the model learns by measuring the value of prediction errors ( cross entropy in this case ), why doesn't it do this by writing internal code that would check the correctness of these calculations, instead of hallucinating? The code is correct and gives the correct answers.
<br /><br />
<b>BUT AT THE SAME TIME THE MAIN SENSE OF QUESTION ( CORE ) AND WAY TO BUILD ANSWER, MODEL UNDERSTANDS AND BUILDS WELL, even sometimes model has no idea what is writing about and is very misleading.</b>
<br /><br />
Already on the 4th model, I noticed that some topics were generated completely randomly, just to generate something. I analyzed structures, headers, etc. from EDK2 repo (https://github.com/tianocore/edk2), and some of them gave very bad results. Even when I gave them the correct headers and structures, model mix them up badly, which later made me lose track of the analysis when model gave me the wrong answers and instructions.
<br /><br />

I only have the code that was generated at the end which is working correctly, it runs and shows the correct result for these questions

```
# pip install tiktoken
import tiktoken

text = """def fib(n):
    if n<=1: return n
    a,b=0,1
    for _ in range(n):
        a,b = b, a+b
    return a

Compute integral: ∫_0^1 x^2 dx = [x^3/3]_0^1 = 1/3.

Notes: Prices rose 12.5% in 2024 — e.g., $12345.67. Mix emojis 😊🚀 and non-Latin: こんにちは. Random shuffle: qwerty-123_ABC.
"""

enc_names = ["r50k_base","cl100k_base","o200k_base"]
for enc in enc_names:
    try:
        enc_obj = tiktoken.get_encoding(enc)
    except Exception:
        # fallback: for model-specific encodings you could use encoding_for_model
        enc_obj = tiktoken.encoding_for_model("gpt-4") if enc == "cl100k_base" else tiktoken.get_encoding(enc)

    token_ids = enc_obj.encode(text)
    tokens_decoded = [enc_obj.decode([tid]) for tid in token_ids]
    print("=== Encoding:", enc)
    print("Token count:", len(token_ids))
    print("Token ids:", token_ids)
    print("Decoded tokens (as strings):", tokens_decoded)
    print()
```

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/chatGPT-OpenAI-some%20chat%20conversations/277%20-%2012-08-2025%20-%20GPT-5%20-%20Still%20hallucinate.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/chatGPT-OpenAI-some%20chat%20conversations/12-08-2025%20-%20big%20picture.png?raw=true)


<hr>
<br />

https://chatgpt.com/share/682fffc7-4690-8000-bab1-9f76c14840b8 - 23-05-2025 - [ PL ] question about the process of producing synthetic fuel from brown coal by the Nazis in World War II. These factories were in Poland, Police, the picture below shows the building where coal was ground into dust from what I found out. BUT WHAT I AM INTERESTED IN IS CHATGPT'S RESPONSE and the manner in which he responded. But the way it drew this process, even though it's a simple drawing without any technological details...

![dump](https://chatgpt.com/backend-api/public_content/enc/eyJpZCI6Im1fNjgzMDAxZGFkYzVjODE5MThlZjRjNDViMzRmZmViYTI6ZmlsZV8wMDAwMDAwMDNhZTg2MWY0OTM5ZDlhOTk1NmI2ODg4MyIsInRzIjoiNDg1NTQ5IiwicCI6InB5aSIsInNpZyI6ImEyNTgxZDA5ZTdiMzJhNWM3NzYzNmRlOGIzYTI3NGRiM2FlYmEwNmJiMGM0Yjg4OWI4N2JjMWQyN2RhNTJmNmUiLCJ2IjoiMCIsImdpem1vX2lkIjpudWxsfQ==)

https://pl.wikipedia.org/wiki/Fabryka_benzyny_syntetycznej_w_Policach

![dump](https://upload.wikimedia.org/wikipedia/commons/f/f8/Police_fabryka_benzyny_syntetycznej_dron_%281%29.jpg) 
<hr>
<br />

https://chatgpt.com/share/681b9edf-edfc-8000-bf3f-739cd284bc9c - 07-05-2025 - write the 10 greatest achievements of human civilization. Starting with early history. Next, write what could be the next achievement, on 11th and subsequent places to 20. Write the 10 biggest threats and problems of the current times for the earth. And a short solution in 1-2 sentences under each point. [ PL ] 4o - <b>Currently, the "AI" will provide a list of this type, but it will not in itself protect the planet from these threats.</b>
<hr>
<br />
https://chatgpt.com/share/671c7778-d7c4-8000-96a9-73c51cf46531 [pl] - GPT 4o

:) 
