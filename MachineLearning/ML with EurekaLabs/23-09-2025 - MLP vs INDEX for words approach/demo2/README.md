> [!WARNING]
> This demo is just for fun to better understand error measurement in probability calculations.


If I had to find an analogy for this demo, let's imagine we're now opening a stock market chart of any instrument. We're opening it now and betting (fake money) on whether it will go up or down in 4, 8, or 12 hours. With this combination, we only have two possibilities. So we're either right or wrong. But if we add another extra variable, such as price range that is, in addition to the high or low, we also measure the price and we start measuring how much we've made a mistake without data, simply on the fly. Then the margin of error increases relative to whether we're choosing only the high or low. In other words - it's harder for us ( for human ).
<br /><br />
This demo doesn't work exactly like this. This demo is just to show if adding extra data to words can help with prediction.
<br /><br />
Paradigms will change, and what I think about ML ( AI ) today, but fundamentally it is mathematics that decides. The question is whether this approach can improve anything. In this case, categories are added to words, i.e. in addition to letters and word prediction and reducing those that no longer fit the combination, there is also prediction of types, what programming language the word is from, and the context, e.g. TOP 10 where the word occurs scenarios.
<br /><br />
<b>We don't need to wait for end of training loop. You can write letters during training.</b>
<br /><br />
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/23-09-2025%20-%20MLP%20vs%20INDEX%20for%20words%20approach/demo2/word_sys_demo2.png?raw=true)

index.html for demo2 - https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/23-09-2025%20-%20MLP%20vs%20INDEX%20for%20words%20approach/demo2/index.html
<br /><br />
Few steps from training loop

```
[12:25:41] Training started.
[12:26:51] Epoch 1/60 — combined 5.9685 — cls 1.6157 — extras 0.8539, 0.7904, 0.0388, 2.2808
[12:27:18] Epoch 2/60 — combined 4.4866 — cls 0.7521 — extras 0.5731, 0.4951, 0.0354, 2.2814
[12:27:46] Epoch 3/60 — combined 4.2871 — cls 0.7037 — extras 0.5243, 0.4551, 0.0284, 2.2529
[12:28:13] Epoch 4/60 — combined 4.1180 — cls 0.6393 — extras 0.4650, 0.4188, 0.0368, 2.2540
[12:28:41] Epoch 5/60 — combined 4.0876 — cls 0.6504 — extras 0.4585, 0.4039, 0.0224, 2.2413
[12:29:09] Epoch 6/60 — combined 4.0810 — cls 0.6430 — extras 0.4391, 0.4149, 0.0316, 2.2533
[12:29:39] Epoch 7/60 — combined 3.9287 — cls 0.6192 — extras 0.4034, 0.3762, 0.0263, 2.2283
[12:30:09] Epoch 8/60 — combined 4.0133 — cls 0.6381 — extras 0.4529, 0.3872, 0.0218, 2.2276
[12:30:39] Epoch 9/60 — combined 3.8618 — cls 0.5970 — extras 0.4107, 0.3585, 0.0236, 2.2361
[12:31:18] Epoch 10/60 — combined 3.8322 — cls 0.5945 — extras 0.4019, 0.3794, 0.0177, 2.2052
[12:31:54] Epoch 11/60 — combined 3.8126 — cls 0.5846 — extras 0.4064, 0.3662, 0.0256, 2.2114
[12:33:04] Epoch 12/60 — combined 3.8091 — cls 0.5677 — extras 0.4023, 0.3428, 0.0316, 2.2238
[12:33:37] Epoch 13/60 — combined 3.6954 — cls 0.5558 — extras 0.3690, 0.3412, 0.0201, 2.2131
[12:34:13] Epoch 14/60 — combined 3.6918 — cls 0.5444 — extras 0.3708, 0.3295, 0.0278, 2.2144
[12:34:43] Epoch 15/60 — combined 3.7453 — cls 0.5556 — extras 0.3725, 0.3336, 0.0260, 2.2187
[12:35:13] Epoch 16/60 — combined 3.7043 — cls 0.5594 — extras 0.3727, 0.3407, 0.0300, 2.2002
[12:35:50] Epoch 17/60 — combined 3.8009 — cls 0.5751 — extras 0.3946, 0.3688, 0.0198, 2.1864
[12:36:40] Epoch 18/60 — combined 3.7900 — cls 0.5755 — extras 0.4042, 0.3524, 0.0185, 2.1930
[12:37:18] Epoch 19/60 — combined 3.6872 — cls 0.5667 — extras 0.3676, 0.3323, 0.0204, 2.2068
[12:37:54] Epoch 20/60 — combined 3.6311 — cls 0.5507 — extras 0.3541, 0.3231, 0.0210, 2.2003
[12:39:33] Epoch 21/60 — combined 3.6203 — cls 0.5519 — extras 0.3371, 0.3328, 0.0175, 2.1938
```
