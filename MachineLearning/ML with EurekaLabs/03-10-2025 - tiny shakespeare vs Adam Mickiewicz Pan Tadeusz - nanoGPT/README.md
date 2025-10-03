>[!WARNING]
> TODO. Why? Read the first lines below. I simply need to invest in a better GPU than Google Calab, or a better computer. For now, I'll try to run it next time (next week?) on https://lambda.ai/ GPU cluster. To do a full training and check from the inside how it works. TODO.


<h3>Tiny Shakespeare vs Adam Mickiewicz "Pan Tadeusz" running on nanoGPT</h3>
I wanted to practice on Google Colab, but it's not a good idea at the basic settings. With 10M parameters, training even 3200 steps takes about an hour. For me it doesn't make sense anymore. I need to move to a better GPU, for example from https://lambda.ai/. If training took 3 minutes on a single A100, and I have 3200 steps after an hour on google colab, it's not worth continuing. Or invest in a new computer. Google colab is ok, and free, but... (...) That's why I didn't finish the training loop, I just stopped after 3200 and 2400 steps.
<br /><br />
Why Adam Mickiewicz, Pan Tadeusz? Andrej can understand what Shakespeare writes. Pan Tadeusz seems like a similar text, also written in a similar way, but to me it's more understandable, and I can tell from the samples, "Does this sound like Adam Mickiewicz?". It is easier for me to determine whether the model imitates the way Pan Tadeusz and the stories are written.
<br /><br />
TODO. But this will be doable once I upgrade to a better, newer, faster GPU. Where training actually lasts those ~3 minutes, not several hours.
<br /><br />
This was running on the default setting - https://github.com/karpathy/nanoGPT/tree/master
<br /><br />
Original text : <br />
1. https://wolnelektury.pl/katalog/lektura/pan-tadeusz.html<br />
2. https://pl.wikisource.org/wiki/Pan_Tadeusz_(wyd._1921)<br />

<h3>What's in the files</h3>
1. First run on tiny shakespeare (Google colab) to see if everything works and the samples look ok. https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/03-10-2025%20-%20tiny%20shakespeare%20vs%20Adam%20Mickiewicz%20Pan%20Tadeusz%20-%20nanoGPT/Untitled11.%20-%20shakespeare%20ater%203200%20steps%20-%2003-01-2025.ipynb<br />
2. This is a long file because it contains the entire text of "Pan Tadeusz." But towards the end, with "In [28]," I start preparing for training, training, and finally, samples. - https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/03-10-2025%20-%20tiny%20shakespeare%20vs%20Adam%20Mickiewicz%20Pan%20Tadeusz%20-%20nanoGPT/Untitled11%20-%20mickiewicz%20sample%20and%20train.ipynb<br />
3. prepare_mickiewicz.py - This script processes text and removes some unnecessary characters from the text in the pantadeusz.txt file. It then produces out.txt, which is processed as a training dataset. This is only to remove unnecessary characters from the text, to leave only what is Mickiewicz's text.<br />
4. out.txt - This is a dataset that will be divided into train, eval, test.<br />
5. mickiewicz_prepare.py - is the equivalent of this, only I changed the names -> https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py<br />
6. config_mickiweicz_train.py - is the equivalent of this -> https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py<br />
<br />

<H3>Sample after ~iter 2500: loss 0.7487, time 74715.03ms, mfu 0.63%</H3>

Why did I take the same approach with letters instead of words? Because Andrej was right to give such an exercise, a demo. And initially, I want to first see how it performs and reach the point where I can say there's a strong resemblance to Pan Tadeusz, but it's not a copy <> paste, it's actually a model imitating it. But as I wrote above, I need to setup a better GPU. And speed up whole process.

```
Overriding: out_dir = out-mickiewicz-char
number of parameters: 10.65M
Loading meta from data/mickiewicz/meta.pkl...


Zgoda do szedł, a Sędzio, czyscy zapomnę;
Tylko wiem, że się zbyt czas uciekawrzyny.
Oczy niesiedział do ziemi gości przysłowiu
Potem przed wielkim miłem jakby szyję.
Polowanie, Noc, Sędzia wielki mój Sędziemu nas milczenie,

Ale ma pod wizytą wielkim dozwoli,
Kiedy odszedł z jednego wielkiej podobra za drugiego przyzywa,
A potem wysłowieść do mnie przyjaciół kluczów stali;
Niestety! on go od ziemi panie Klucznik z Rymsza.
Spóźno są ze szlachetki wielkiego pierwszy spoziera

Psy utrapionego! Ko
---------------

Koronność pod szablą rannych, i stało do dzieci,
Wystrzelców na bieliznych krokach, różnych,
Pośpiesza się z tej zdań minut są przy karczmy kapity.
Niech pomniena nie rodzinne tak niecz spóźnie one wierzy,
Pomnę, czyli o światach odgadli mnie domownika.

Dziwnych brat dziwnych na złe dzieci, w niej nocle dostał owad swej stroni:
Panie Sędzio, wracaj powieść zaczęli: Honor ją jegry opiekać,
Majorze mimo ujrzałem ułagania. Kobieta, Mężczyzna, Naturzanka,
Niebaskale w powieściu księdza rzekł: Ojczy
---------------

Kiedy był się podbijem się z obławy.
Jak stolnik widzi wejmie zamknięcia

Przy stołu oczy wojnnych, ale pomnij zgrzyta.
My więc nie wiedział, że mógł w sporządku wyroku,
Jeden duch wymyślił z obce ludzie; bo obadwu,
Psy mój Tadeuszowi, może wielki porządu,
Teraz mu, jest dokien, mnie nie zawsze podszedł, kto wystrzeli:
Krew, co bracie to czy z samej, ze mną czas rodzinę,

I wyciemają choć się podróżnie karę nad nim gadać.
Gdy się to już sobie radził przy zamek dmucha,
A pan zamek pan Sędzia z Ró
---------------

Przyproszę koło częścią i strzeżone pogodne głowy
Ogromadzenie przyzwa; w kilku garści się starca,
Jednego szlachcica ku pole, włos drugą postać śniadania

Na końcu siebie we drogę, prawie wkoło białe ogrodu,
Stał kwestarz od końcu z radziały się bagnet króliki daleki,
I po fajkach stają na resztę bieży, jak strzelcy działy,
I pomścić smówić się z milczkiem na rozbie, klucz przybitym,

Wzdłuż się nazwisko w kapelu i uroczył głowę.
Panie Sędzio, niech bogata, niech nie pozwalał
Kobieta, Mężczyzna
---------------

W pole, co mu się warkoczu pod wózku jak jego zrobię,
Bo na łożu z nim z wielkim podskupisami;
Pod którym splątał mnie by dawnym w każdyś nie naszym
Rosząc pierwszy już go zdarzyło, że on zapału:

Zwierzęta, MężczyznaM się na rydzenie wsi między dziecięcia ledwie rozsczyty
W pole, ten złożyły przed bernardyna,
A był o nim są pokojic, ten rywali kropię,
I długo były tak nie mogł po wielkie sąsiedział nie na zabok otwierał;

Moda, Obyczaje, Sława, Sława, Kobieta, Mężczyzna, Musiarie, DrzewoNie był
---------------


Kiedy był mu na czyli obrazu.
Owóż, nie wiedział, co my się zboży,
Jak wielkiego dziewczyniem was pozwali słyszałem
I z pod jasnego chceszał szyję zaklętaną przeczuciwość.
Dama, Dziedzictwo, PanOwa jest pod proces, domy zaprasza.

Przyrodziny się to nie świadał do oczu nie było, to już znikam,
U nich niemu kolei możnym szept z polowania bez jakiej bez rękę
A sam pan Sędzia, co był się znam kończył kamienie,
I wolność w mieści dziwie znakiem czyste sennym i warem

I z radością gospodarstwa, jak 
---------------

On, wszyscy przyjść tego wystrzela.
Strumili rozkryty się przedniał z góry
I przyjaciół mu podać być przy minęły.
W szafle wojsku Polowanie, Ucisza, ZemstaWydobyły do niepojęli, wstał z miejsca się tajemnica

I potrząsając słuchach, z góryczkami nie było dostania. 
Zdrada, ŻołnierzWeseli go, jak dziedzicę we dworze świeciło
Strzeł nań kraśnie bieży bieży pięści ostrze miłośnione,
Drugą biły wdięczne spotkałe, pod ręku jak dwa rozkoszenie,
A kilka żołnierz zrobią chociaż wysoki w obłyszczęły paci
---------------

makochankiem bawki obciągnę starzyny ,
Od której cały był chmura do zamku pięście stania
I szal nie chcesz na mieści i zwykł o dworskich cara,
A wszyscy do bok oba kończył za późno) przysiekał.

Hrabia zatknął, że się począł z wielkim oczy się macierzyć.
Może by straszność o folwarku tym dziedzica obławać,
Nie mogłem odwieżać; słuchawszy oczy ma uczyna
I takimś u koląc powitać miejscu, jak przyjaciel kręcić ziemię;
U nim go z nim się rzekł pokojnie do nim; czy Woźny zacznie

Czyli o tym przezróż
---------------

Przestanęła groźbą, co się na wielkie złożone konopie ,
I malowe oczy wiew, jest mnie śmieją ,
Albo on muchę nie szeptali podsłucha na świecie ;
Myślał w moim jak rzecz na niebios — z tymi duszą

O miejscu, i po szalach prawo. Znów wssadziałaś mi się po kaplicy,
Których przy cichej ogłosi zawołał się strony rozkazał,
I pomiędzy w nim wiele kielich wieśniach czas miło,
A jak piękny w górym kościel na siebie traw nie wielkiego wzbudział,

A pierwsi w myślach wielkiej stawiłem odstąpił,
Ustawiwszy 
---------------

Od słońca, jak peszczone dostatniego nagle upada.
Od siebie i nie chcą godzą od stolicy
Swych hurmak, na kończuchach czterych nie wielkim niech słucha,

Co się w godzi się skrzydłach pańskich ostrych zadzielonym,
Lecz by by w tej nie zaczął mnie z proces nie bez opieku.
Nawet on. Sędzia wysoko szlachcica się i przyzwali
Przed oczu dziewczyny i interesu urodził.

Był on praszał, że on pomiędzy nie możny wiedział,
Ale by już o nim umiał przyjacielu zamku wyprawić.
Niech , cóż był nie był do tym so
---------------
```
