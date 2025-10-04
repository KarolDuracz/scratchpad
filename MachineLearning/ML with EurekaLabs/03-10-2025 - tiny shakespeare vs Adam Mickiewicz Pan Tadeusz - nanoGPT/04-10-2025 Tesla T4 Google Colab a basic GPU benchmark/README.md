<h2>Basic test to have a comparison - 04-10-2025 - </h2>

Basic benchmark comparison on the smaller model to have an initial comparison with better GPUs and the 10M and more nanoGPT model

test 1 - https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/03-10-2025%20-%20tiny%20shakespeare%20vs%20Adam%20Mickiewicz%20Pan%20Tadeusz%20-%20nanoGPT/04-10-2025%20Tesla%20T4%20Google%20Colab%20a%20basic%20GPU%20benchmark/Untitled12%20-%20smaller%20model%20and%20basic%20benchmark%20test%20for%20T4%20-%2004-10-2025.ipynb
<br /><br />

<h3>Profiling</h3>

block line [16] - %%writefile benchmark.py><br />
listing [17] - !python benchmark.py --duration 60 --profile

```
Device: cuda, device_type: cuda, dtype: torch.bfloat16, use_model_requested: False, have_gpt: True
CUDA device name: Tesla T4, total_memory: 14.74 GB, major: 7, minor: 5
Using synthetic workload for benchmark (no external model required).
Warmup: 10 iterations
Running microbenchmarks
Forward-only iter 4/20, sample loss: 0.0000, iter_time 405.62 ms
Forward-only iter 8/20, sample loss: 0.0000, iter_time 402.56 ms
Forward-only iter 12/20, sample loss: 0.0000, iter_time 409.99 ms
Forward-only iter 16/20, sample loss: 0.0000, iter_time 407.85 ms
Forward-only iter 20/20, sample loss: 0.0000, iter_time 423.37 ms
Forward-only: 20 iters in 8.14s, tokens/sec: 30174.7
Forward-only: GPU mem allocated: 42.1 MB, reserved: 6954.0 MB; peak_alloc: 3882.1 MB, peak_reserved: 6954.0 MB
Backward-only iter 4/20, sample loss: 0.0000, iter_time 597.05 ms
Backward-only iter 8/20, sample loss: 0.0000, iter_time 604.91 ms
Backward-only iter 12/20, sample loss: 0.0000, iter_time 596.51 ms
Backward-only iter 16/20, sample loss: 0.0000, iter_time 590.81 ms
Backward-only iter 20/20, sample loss: 0.0000, iter_time 598.54 ms
Backward-only: 20 iters in 11.84s, tokens/sec: 20765.0
Backward-only: GPU mem allocated: 85.8 MB, reserved: 6954.0 MB; peak_alloc: 3890.2 MB, peak_reserved: 6954.0 MB
Full-train-step iter 4/20, sample loss: -0.0000, iter_time 581.73 ms
Full-train-step iter 8/20, sample loss: -0.0003, iter_time 596.06 ms
Full-train-step iter 12/20, sample loss: -0.0036, iter_time 607.75 ms
Full-train-step iter 16/20, sample loss: -0.0349, iter_time 607.84 ms
Full-train-step iter 20/20, sample loss: -0.2871, iter_time 617.11 ms
Full-train-step: 20 iters in 12.33s, tokens/sec: 19936.4
Full-train-step: GPU mem allocated: 153.3 MB, reserved: 6954.0 MB; peak_alloc: 3957.7 MB, peak_reserved: 6954.0 MB
Starting torch.profiler for ~60s; traces saved to ./bench_log
Profiler loop step 10, elapsed 6.3s, last_loss -40.2500
Profiler loop step 20, elapsed 12.5s, last_loss -1312.0000
Profiler loop step 30, elapsed 18.8s, last_loss -9408.0000
Profiler loop step 40, elapsed 25.1s, last_loss -46848.0000
Profiler loop step 50, elapsed 31.2s, last_loss -225280.0000
Profiler loop step 60, elapsed 37.3s, last_loss -991232.0000
Profiler loop step 70, elapsed 43.4s, last_loss -4063232.0000
Profiler loop step 80, elapsed 49.4s, last_loss -11403264.0000
Profiler loop step 90, elapsed 55.4s, last_loss -41943040.0000
Profiler run finished: steps=98, elapsed=62.77s

Top CUDA ops by cuda_time_total:
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  Total MFLOPs  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                           aten::matmul         0.05%      31.686ms         0.29%     173.639ms      88.591us       0.000us         0.00%       69.000s      35.204ms           0 B           0 B      37.65 GB     -77.77 GB          1960            --  
                                             train_step         0.00%       0.000us         0.00%       0.000us       0.000us       60.092s       100.04%       60.092s     613.185ms           0 B           0 B           0 B           0 B            98            --  
                                               aten::mm         0.21%     128.845ms         0.29%     174.080ms      39.474us       50.469s        84.02%       50.469s      11.444ms           0 B           0 B      53.88 GB      53.88 GB          4410  123115237.540  
                                             train_step         1.25%     759.162ms        99.17%       60.169s     613.966ms       0.000us         0.00%       43.872s     447.671ms           0 B           0 B           0 B    -573.60 GB            98            --  
void magma_sgemmEx_kernel<float, __nv_bfloat16, __nv...         0.00%       0.000us         0.00%       0.000us       0.000us       37.007s        61.61%       37.007s      22.213ms           0 B           0 B           0 B           0 B          1666            --  
                                           aten::linear         0.03%      16.285ms         0.51%     309.446ms     105.254us       0.000us         0.00%       16.095s       5.474ms           0 B           0 B      53.56 GB           0 B          2940            --  
       autograd::engine::evaluate_function: MmBackward0         0.05%      30.028ms         0.32%     194.157ms     132.080us       0.000us         0.00%       13.439s       9.142ms           0 B           0 B     -23.60 GB     -49.92 GB          1470            --  
                                            MmBackward0         0.04%      26.804ms         0.27%     164.129ms     111.652us       0.000us         0.00%       13.439s       9.142ms           0 B           0 B      26.32 GB           0 B          1470            --  
void magma_sgemmEx_kernel<float, __nv_bfloat16, __nv...         0.00%       0.000us         0.00%       0.000us       0.000us        9.278s        15.45%        9.278s       5.569ms           0 B           0 B           0 B           0 B          1666            --  
void magma_sgemmEx_kernel<float, __nv_bfloat16, __nv...         0.00%       0.000us         0.00%       0.000us       0.000us        7.778s        12.95%        7.778s       4.668ms           0 B           0 B           0 B           0 B          1666            --  
                                            aten::copy_         0.08%      49.476ms         0.15%      92.445ms      24.188us        3.831s         6.38%        3.831s       1.002ms           0 B           0 B           0 B           0 B          3822            --  
                                               aten::to         0.01%       8.678ms         0.27%     166.784ms      42.547us       0.000us         0.00%        3.786s     965.916us           0 B           0 B     317.66 GB           0 B          3920            --  
                                         aten::_to_copy         0.06%      35.413ms         0.26%     158.106ms      42.456us       0.000us         0.00%        3.786s       1.017ms           0 B           0 B     317.66 GB           0 B          3724            --  
                                              aten::bmm         0.03%      19.642ms         0.04%      26.230ms      44.609us        3.594s         5.98%        3.594s       6.112ms           0 B           0 B      11.48 GB      11.48 GB           588  11364483.465  
      autograd::engine::evaluate_function: BmmBackward0         0.01%       3.829ms         0.05%      28.324ms     144.508us       0.000us         0.00%        2.314s      11.804ms           0 B           0 B      -4.02 GB     -11.48 GB           196            --  
                                           BmmBackward0         0.01%       4.060ms         0.04%      24.495ms     124.974us       0.000us         0.00%        2.314s      11.804ms           0 B           0 B       7.46 GB           0 B           196            --  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us        1.507s         2.51%        1.507s       7.690ms           0 B           0 B           0 B           0 B           196            --  
                                          aten::one_hot         0.00%       2.831ms         0.03%      16.816ms     171.587us       0.000us         0.00%        1.366s      13.942ms           0 B           0 B     294.00 GB           0 B            98            --  
                                            aten::fill_         0.00%       2.909ms         0.01%       5.905ms      30.127us        1.364s         2.27%        1.364s       6.960ms           0 B           0 B           0 B           0 B           196            --  
                                            aten::zeros         0.00%       1.213ms         0.01%       7.130ms      72.757us       0.000us         0.00%        1.364s      13.918ms           0 B           0 B     294.00 GB           0 B            98            --  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 60.673s
Self CUDA time total: 60.067s


Top CPU ops by cpu_time_total:
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  Total MFLOPs  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                             train_step         1.25%     759.162ms        99.17%       60.169s     613.966ms       0.000us         0.00%       43.872s     447.671ms           0 B           0 B           0 B    -573.60 GB            98            --  
                                             aten::item         0.01%       5.735ms        97.17%       58.956s      19.406ms       0.000us         0.00%     191.447us       0.063us           0 B           0 B           0 B           0 B          3038            --  
                              aten::_local_scalar_dense         0.01%       6.038ms        97.16%       58.951s      19.404ms     191.447us         0.00%     191.447us       0.063us           0 B           0 B           0 B           0 B          3038            --  
                                  cudaStreamSynchronize        97.15%       58.942s        97.15%       58.942s     601.452ms       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B            98            --  
                                           aten::linear         0.03%      16.285ms         0.51%     309.446ms     105.254us       0.000us         0.00%       16.095s       5.474ms           0 B           0 B      53.56 GB           0 B          2940            --  
       autograd::engine::evaluate_function: MmBackward0         0.05%      30.028ms         0.32%     194.157ms     132.080us       0.000us         0.00%       13.439s       9.142ms           0 B           0 B     -23.60 GB     -49.92 GB          1470            --  
                                               aten::mm         0.21%     128.845ms         0.29%     174.080ms      39.474us       50.469s        84.02%       50.469s      11.444ms           0 B           0 B      53.88 GB      53.88 GB          4410  123115237.540  
                                           aten::matmul         0.05%      31.686ms         0.29%     173.639ms      88.591us       0.000us         0.00%       69.000s      35.204ms           0 B           0 B      37.65 GB     -77.77 GB          1960            --  
                                               aten::to         0.01%       8.678ms         0.27%     166.784ms      42.547us       0.000us         0.00%        3.786s     965.916us           0 B           0 B     317.66 GB           0 B          3920            --  
                                            MmBackward0         0.04%      26.804ms         0.27%     164.129ms     111.652us       0.000us         0.00%       13.439s       9.142ms           0 B           0 B      26.32 GB           0 B          1470            --  
                                         aten::_to_copy         0.06%      35.413ms         0.26%     158.106ms      42.456us       0.000us         0.00%        3.786s       1.017ms           0 B           0 B     317.66 GB           0 B          3724            --  
                                       cudaLaunchKernel         0.22%     135.009ms         0.22%     135.009ms      10.934us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B         12348            --  
                               Optimizer.step#Adam.step         0.09%      55.534ms         0.18%     111.689ms       1.140ms       0.000us         0.00%     288.081ms       2.940ms           0 B        -392 B           0 B      -3.23 GB            98            --  
autograd::engine::evaluate_function: ToCopyBackward0...         0.02%      12.944ms         0.18%     107.095ms      64.283us       0.000us         0.00%      92.318ms      55.413us           0 B           0 B       1.21 GB      -9.09 GB          1666            --  
                                        ToCopyBackward0         0.03%      16.795ms         0.16%      94.151ms      56.513us       0.000us         0.00%      92.318ms      55.413us           0 B           0 B      10.30 GB           0 B          1666            --  
                                            aten::copy_         0.08%      49.476ms         0.15%      92.445ms      24.188us        3.831s         6.38%        3.831s       1.002ms           0 B           0 B           0 B           0 B          3822            --  
                                                aten::t         0.04%      25.489ms         0.10%      59.751ms       8.239us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B          7252            --  
                                          aten::reshape         0.03%      18.724ms         0.07%      43.031ms       7.984us       0.000us         0.00%      44.485ms       8.253us           0 B           0 B       1.72 GB           0 B          5390            --  
                                    aten::empty_strided         0.07%      42.594ms         0.07%      42.594ms       8.049us       0.000us         0.00%       0.000us       0.000us           0 B           0 B     320.89 GB     320.89 GB          5292            --  
                                        aten::transpose         0.04%      27.009ms         0.06%      39.139ms       4.992us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B          7840            --  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 60.673s
Self CUDA time total: 60.067s

After profile: GPU mem allocated: 153.3 MB, reserved: 6954.0 MB; peak_alloc: 3957.7 MB, peak_reserved: 6954.0 MB
Done. If you ran with --profile, open TensorBoard on ./bench_log to inspect flame graphs and step traces.
```

<h3>3 tests for shakespeare_char 0.80, ~3M and ~6M params models to find what loss looks like and basic info</h3>

<b>First</b>
<br /><br />
[1] - [6] - shakespeare_char launch><br />
[7] - !python train.py config/train_shakespeare_char.py  --device=cuda --block_size=32 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128

```
# just to have a comparison, more info in the .ipynb file - number of parameters and loss for last step
number of parameters: 0.80M 
iter 5000: loss 1.8097, time 2196.25ms, mfu 0.03%
```

[8] - sample for 0.80M
<br /><br />
<b>Second</b>
<br /><br />
[11] - !python train.py config/train_shakespeare_char.py  --device=cuda --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=256

```
number of parameters: 3.16M
iter 5000: loss 1.5617, time 2266.47ms, mfu 0.27%
```

<b>Third</b>
<br /><br />
[13] - !python train.py config/train_shakespeare_char.py  --device=cuda --block_size=64 --batch_size=24 --n_layer=8 --n_head=8 --n_embd=256

```
number of parameters: 6.31M
iter 5000: loss 1.3742, time 5142.22ms, mfu 0.53%
```

[14] - !python sample.py --out_dir=out-shakespeare-char

```
Overriding: out_dir = out-shakespeare-char
number of parameters: 6.31M
Loading meta from data/shakespeare_char/meta.pkl...


Clown:
Repence, and is not the disposities of a
virtuous called
My son that use a bed by the root.

First Citizen:
O, my lord, or some for the covert of the beggins
The king late may revenge and the news is wars
Well and peace the mild with a children hew you:
This I can patientle home.
I think your oath of your pity sound me,
And as I shall strong of his guilty dangerous fire;
And he must not may poor for dear?

KING HENRY VI:
Hark you and advance the heart of the king,
And I will change the w
---------------

MenENIUS:
And such seems that hath hate myself and fortune
Will and sweet blind with welcome, whom come something
To glory this poor appears.

LADY ANNE:
How shall discoversed might commanded:
He shall be gone and your grace; as they have done
With our courtain bearing the country and me with self.

CLIFFORD:
No, my lord, come I say; let it him the name,
That tell me the offen so the noble eyes and approache with the complain.

ESCALUS:
Hark no heart of Lord Bartham it, would I will;
To know her
---------------

Messenger:
The peace, I have no more dead convertain,
And take the countent such all not rest mine
Of feeding in in her world the kings of deceived
That that wouldIng my admition and love thee.
Then thou lament therefore sing but made my hand;
Though is the determy of Hereford,
That lands grant of the world a defend,
So much he call'd me be love in state,
All triumph'd the king of my best defence,
Cry the power of victory whose was headen and smalls,
The play well plains yet of Roman:
But, he ha
---------------

The state of your king,--what I will die
By the king, scarce your courtents upon his parts.

DUKE VINCENTIO:
Why deserves now, what he the true of him?
Thus I will are a godding become again.

KING RICHARD III:
I have blacknothed the wife in me father,
And I come, to one that, would they weak thee with fair
Dispatch'd with the peeroner like a spurp from his
traitor of the queen and sword, promised my hears
That take the life of on the Richard, to said shall
I find thee such a hearther of a man.

---------------

Be every believe, her lost the very time
Are fellow on the face of many be an haunt
Which thou rad upon the kings of the land,
And I were advertain as regotten of it.

AUFIDIUS:
In the name of mine word.

LADY GREY:
Why sleep the greaters of my lord;
And she chamberous lambs unto him.

GREMIO:
Hear adown him when it not are upon it lady's.

Clown:
Let's himself I speak, the sea of the obey;
And still then he are not for move be a thing.

QUEEN ELIZABETH:
Where is sleep not tears will furthes for
---------------


MENENIUS:
He is the contented throne being in the gentleman,
And they drawn with me not ever of the husband;
Who are the powers of his friends so see to the fees,
That I have got me in creeted with it is no oath,
Which is my son men as reason well for a mistress,
And this short, here but the proper of the Lancaster's death?

KING RICHARD III:
Then are the king: to me, and not thou dost had
distroop the Coriolanus! therefore my life?
And how the gods of head of the death worses
I will and little
---------------

Shall be well thee of fairers, and shall,
I know the news of an thought of the field:
And susper at your grace fault in storms.

BRUTUS:
Thou dost you throw time to have our king,
But all right, these love of the sleeper married wine
To disour.

KING RICHARD III:
What for the rise was from this tempture.
How now shall death.

BRUTUS:
I do not find, and trouble lives a while I will
have faced chysing with a soldier, and your brace
Oaths in joy wounds not call me to sure.

ANTIGONUS:
All that I bu
---------------

lords how dreams of the execute of mine heart
And the bosom that have and the reason.

AUTOLYCUS:
For was all love my never grow:
But it be a true in a lessard and land,
So bear'd his and death--bower and word
More power: and summons unbefore a sweet love,
And never treater tends the still reportor
That you strike so after to be the voice,
Whilst every you content of the vengeance to do
And the please of your house: what for the brother
Since upon the melar, and she are men had
with a word to be
---------------

Stand and the ready be fan made mean's past and mole,
Being in the crowns being of fourtune's sons,
Or Rome is the loss-lips of shape sound is the good
In the face of Bolingbroke of Juliet'st queen.

SICINIUS:
In Clarence, if we all reason with the sun?

DUKE OF AUMERLE:
Proud of your sight very hastings of hath carmand
To do put thee to do his head upon the father:
And it be to truth all the giverness of be so mon
Both loves and her is the bonder. As we shall be through.

QUEEN MARGARET:
Prithe
---------------

How charged traitors, and himself and speak,
If I am house the deceiver of mine own in
them; if you know not call the shape tell me,
Do think your better proper that you sprite
The red violent persuades that would have your sentence.

QUEEN ELIZABETH:
Go me, and yet for
And love-diwn beyond, and are the drum.

RICHMOND:
I know not command to his grace to me of your
sail mercy, and we may your since upon him a sin
sigh either dead unto of his country's fearful it.
What,' good common that the serv
---------------
```



<h3>Mickiewicz dataset</h3>

https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/03-10-2025%20-%20tiny%20shakespeare%20vs%20Adam%20Mickiewicz%20Pan%20Tadeusz%20-%20nanoGPT/04-10-2025%20Tesla%20T4%20Google%20Colab%20a%20basic%20GPU%20benchmark/Untitled12%20-%20mickiewicz%20test%20smaller%20model%206%20M%2004-10-2025%20google%20colab%20T4.ipynb
<br /><br />
Start from listing [23] - saving text, preparing a model, etc. Same as in the first test, same commands.
<br /><br />
[36] - !python train.py config/train_mickiewicz_char.py --device=cuda --block_size=64 --batch_size=24 --n_layer=8 --n_head=8 --n_embd=256

```
number of parameters: 6.32M
iter 5000: loss 1.6849, time 5171.02ms, mfu 0.56%
```

Sample for ~6M model - [37] !python sample.py --out_dir=out-mickiewicz-char

```
Overriding: out_dir = out-mickiewicz-char
number of parameters: 6.32M
Loading meta from data/mickiewicz/meta.pkl...


Zgodził się do karczmę od cudniejszym pory,

W Rózecz koniec i pierwszy i gromadą swoje;

A musiał się do niego ode szyję po łopie,
Którą bardzo w szlachcic bracie mówić do kobrę?
Czasem za Woźny nowo się wziął po mimo szlachcicie,

Nie więc chwycił tym w zgody nowe odszedł w błekkiem
I rzucił uklepnie się; pierwszy za wieczerzy

Jak zadziwił się krzyknął: 
On leży wołał się, spojrzał nagle szlachcica!
Tak w szlachtą: już nie często ze sądu serce zdoby,
Jak w jedno lata, wolność chwila nie wsty
---------------

Korona silnie szlachcic po całej paszych i stole.
Szczerego wiszedł trafia ku na wodzów zwycięstwem
(I gwiazda powiat minurzyja nie możej wiąże.

Więc co mnie wybawiał: dawny, tak niech spotkanie
Pomiędzy nie domowała, żeś patrzył jarzyna.

Przerwał Sędzia, dostał Klucznik i czasem nie przysiadła.
Daje strona, co posłania do kiwatach myśliwi
Kura w objąwskich naznajwaśnie wyskropnie się wodził.
I wyjedział, chociaż pod ustach tak więc czy spod się 
Darmo, cóż się w środek, u kształt rzecz po pap
---------------

Kto mogłem całem gęsto pan Forta, panie miał w środku. 
Na błysnął Sędzia — to po rzecz nasz wnet tym wyprawiając.
Zdrada wielki Brzytewki sadzą mógł chrzciciel w Gospodarza;
Jak mu na czym wesół, ludzie może bez miałem,
A się od nowym z asami pan Ryków uchwał,

Ozgramał w księgim mieszał, albo podskoczył.
Sędzia w Soplicowie: Tadeusz nieco moje się nie kapicia,

Odbijać, że szlachta choć to nie wieczerz przyjdzie w kapeluszach.
Nie już został na rozpieczenie mówiąc gródy,
Grządki bernardynie i 
---------------

Porządku pokoju, znak szereg rozmowym głowy głowy
Ogłuszczył się raz chrzcicieli i pod srebraku.

Czy dzisiaj za dziedzica się, w albo podarty;
Oto na śmiecie im sam skrzydeł się wypisnęła.
Jakby tak, chce mu Soplicowie ich świeciły,
Że miał na złość ciemnie mówić całkiemu się opieku;

Tak był ważny znowu, co przepraszać się małżone.
Nam te Sędzia, gdzie żyju się był zalono. 
Na ot zapuszcznik, więc widział księżyc szczęśliwym 
I Hrabia na gościny rzecze wojnym procesy,

Ustawił się o charty, ja
---------------

W polskich mu swą i rąk nie zdrada trudnie,
Każdy na ognie słomu z nim zawiesz uderzył,

Nad Polskiemu tylko bawił na Moskala pan Ryków!
I ona wielkiej nie złote jej przecież poły!
Miał na rewon przybić Żyd poznał na domowym,
Tymczasem się może wyzwał się na dziele,
Ale znajdzie jak zawsze ten zakończył ziemię:
A jeden broń spojrzał, ten równie trzysnął kołtunek 
Lub i nie mogłe wielki, w sąsiedział się na fartowy.

Jest pomiej pięknie, jak pierwszy wzglądki słuchami,
Pogromny na zakryć do okien
---------------


Kobieta, Brzytwa, Klucznik, Ach, wyrucza i chybił nie przezboże!
Ja zdrada; Ale może wysięgi pan ustać na słysza.
Zawsze gromadził Rejent stary w tym szerokiem trupie,
Czy pamiętami jak stronia nam jeszcze potrzebnie,
Poznał się w poły swojej strzelcy na wyroku
Wojski rzuca żołnierskich; rzekł się bubi, spokojny,
Podobna wzglądam jak przybiegł walcze, pędzie się panie.
Czy też Rejzan — na koniec pastał — skrzecznik zmówił?
Na dziedzicami ludzie przymyślił ten wyjechał;

Bo to go skupnie za nim 
---------------

On, co mówiła się grunku wystawioną.
Wie, że rozmowa ostatnik, niech kłowam w końcu stary,
Że do mu zabawił najprzywiedzieć podobne do Podkomorzy.

Nie może wodzie widział, że się chwytają nim nowe
Nie powstała na wielkiego wspomadł zwierzał się konopie:
I to się szlachta nie stało z sędziemu jak mądroga,

I strzelca zamek się dołem w kraju widziałem,

Gdy na Sędziego mocny przed ładości drogę:
Rozmowanie, pędzono wszystko, co w ziemie, bok pisardowa —
Warte obszczacjon wysok panowie:
Radziwia s
---------------

Tako kiedy się w proży karczma, znowu był dzihem, we walczył,

Na niéj przybósł się ciebie szanu pacie było:
I więc za dwok koniąc mu jak ostrzecie szyję,
Kiedy na końcu Robaka zwano szkółka, a z przy szeptaki
Kobiet kochanka w koń tyle zgodnie stronny .
Może do stary fartan forterkusty sadszy,
Na okne ma rzucił jak obraz hustki, kiedy ze szpadania.
Przy to połowania się w tym mogu poranie.
I panie Sekuszki — widział Klucznik — wołała radość niebiosą —

Tam w pokrzyczach: rzekł Hrabia — czy ni m
---------------

Przesłonęła groźnie Soplica w wielkiej zrobie!

Czasem, lecz Podkomorzy twój wstrzyma światem,
Ale to na cię nie sztuki. Wiedząc ocałym się jask nie wielką
Ogniadał węgi, na polowaniu i piasunką

Z miądzych ziemię, że olczyzny się owaszał;
Na sto sztuka jak dalej szaraki, zwani jedynek,
Po awnie mocnym ludzon i zdawał i ,
Choć w takiem głębie powszeciły z mych wielkim,
Wyjaciel się jak mu młodzieńca się pacierze
Podobne mocny zatrzymaje królewski,
Tyle na wsi wielkie posłecznych na błyskach wstr
---------------

Od głowę i w suwesor i panie trzeba

A gotowa go za Birba w końcu chwilę szlachcica,
Tymczasem była w tyłu w dawnych ciebie wamię.
Ale Sędzia w Pauciej drugich — to rzekł Soplicowski — wszak to Jacek za wielkim kaszałem
Zakończył się zadrżami nie słuchać nie bezpiechy
Podawać w mój szlachta bernardyn Książęciem

Służy, ZgródBył do dziewczyny i i poginnek;
Dwie wydarto z naszej zaścionych z daremnicę słych,

Sztuka się z księgałów obiadwadzieńco zwycięzce,
Po czarnymi się wzniósł pierwszy śród si
---------------
```

```

