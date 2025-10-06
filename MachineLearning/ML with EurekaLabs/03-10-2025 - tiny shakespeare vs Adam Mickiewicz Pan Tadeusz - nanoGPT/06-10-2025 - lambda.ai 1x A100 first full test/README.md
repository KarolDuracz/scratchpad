<h2>lambda.ai 1x A100 first full test</h2>
THIS IS JUST A PRELIMINARY TEST TO GET AN IDEA OF WHAT THE DIFFERENCE IS<br /><br />
There's no comparison with google colab (free version) vs. lambda (1x A100). It's easier, faster, and better. I specifically tested it on a 1x A100. Like in the previous demo, I wanted to see if it actually lasts 3 minutes. If you look at the .ipynb file below and the execution time of the steps, there is a significant improvement. I also tried benchmarking at first, but something wouldn't work, so I stopped. Then Shakespeare and Mickiewicz. That's me on the previous demo from a few days ago.<br /><br />

https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/03-10-2025%20-%20tiny%20shakespeare%20vs%20Adam%20Mickiewicz%20Pan%20Tadeusz%20-%20nanoGPT/06-10-2025%20-%20lambda.ai%201x%20A100%20first%20full%20test/Untitled-mickiewicz-A100.ipynb <br /><br />

> [!IMPORTANT]
> In .ipynb file I see now that it cut off. But the image below shows that it finished training after 5000 steps and generated samples. It's nothing. I have something anyway.

screenshot from the last test with samples for Mickiewicz

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/03-10-2025%20-%20tiny%20shakespeare%20vs%20Adam%20Mickiewicz%20Pan%20Tadeusz%20-%20nanoGPT/06-10-2025%20-%20lambda.ai%201x%20A100%20first%20full%20test/test%201.png?raw=true)

Cost - 0.67$ ( USAGE 0.5 hr | RATE $1.29 / hr | SPEND $0.63)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/03-10-2025%20-%20tiny%20shakespeare%20vs%20Adam%20Mickiewicz%20Pan%20Tadeusz%20-%20nanoGPT/06-10-2025%20-%20lambda.ai%201x%20A100%20first%20full%20test/test%202.png?raw=true)

Just like before, I copied the file to my local drive and ran it on the CPU. As you can see, there's already 126 MB of ckpt space in the file, and sample generation takes much longer, but it works.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/03-10-2025%20-%20tiny%20shakespeare%20vs%20Adam%20Mickiewicz%20Pan%20Tadeusz%20-%20nanoGPT/06-10-2025%20-%20lambda.ai%201x%20A100%20first%20full%20test/sample%20on%20CPU.png?raw=true)

<h3>Sample</h3>

```
Overriding: out_dir = out-mickiewicz-char
number of parameters: 10.65M
Loading meta from data/mickiewicz/meta.pkl...

Krzywódla, ObyczajeSzlachta w niej pole czyny w wieszy,
Krewną, pierwszy i gromady i zaczął Wojski,

Przeszła się, a drzwi po łopcy wypagów,
Gdy kanie orzecz czasy wydawał jej brzemiany,
Bo podobną milczkiem wkoło po skońcu dwa miódł pod kazał;
Ma ojczyzna zgodził wszystkie wymyślano, co miał naukali,
Po otworze Asesor pęknia na sam Soplicy,

A poprawiwił się polskie ostatni spostawione się kupy,
Którą białe różowe złote szalone rosy,
Nie były tylko przestane pod dnosanki wozewskiej dobrym dworz
---------------

Koroniony ponczas wody,
I z nimnych pioruntów. Wojna! Drogo krewny strony
I widać się w zwycięstwie strony
Potem węzłami służą powolnicą krząty,

Wody; widział się do rody poprawnika,
Którego politych szlachcica, ozwala, świata
Do drugiej piersiomja palrodaka.
Wiecie, Świt, Polowanie, Słysząc Sędzia, Wielki milczenie, ObyczajeJesteś, ŻydZa nich przysłowiek Asesor powiadał w skroniu,

Który na zamkowieść bole trzechodził. 
Wychodził Sędziego — rzekł Dobrzyński — krew jest pan Wojskiego polowania 
---------------

Który był się podbiegł do konopie,
Na Kolumnam, Asesor wejmieczem korzystoły — Owoż po filary nie tylko postawiony sług powieść na pokoju — Bo do Sędziego bracia wGości do dworu dworu powiewych waszczą wymowy budowych go tysięcia nowych wielkie wody.

Ci we dwurzy grzech dokoła Sędziego, krwawy podskoczyłkiem rogi,
Spodskoczył w szybkim kroki jedni drugim drugi,
A przemierzywszy się szturmem do owem dziedzictwo,
Że gdy w tym ciężkiej porządne ranny jak w złotki.

Chciałem jeśli w położę politews
---------------

Porwany w placu czas armijonych wielkich,

Ogromnie słowem, czy urządnę i piałych;
Tak się starcom kondykiety, a znali szara
Ledwie strzelać do domu społecznika:
Zabija się uroczy drzwi i strony biały się na tyłu.
Próżno za świeciły się milczki z bagnanek szyję do rodu,

Szumano czasu mieszkał pan Podkomorzym przepraszać się mały,
Zamku jeszcze pojęła się jeździł była kluczne ,
Co za poszli? Wiesz o wiesz woli Hrabiego kończyć.
Jeśli po cóż ojcze szlachcicie? gdzie nie panny, czy kiwaj!

Mam bra
---------------

W pułku ciemnych liczkich wiedzi, zapłaciasków,
Gdzie za nogi miał jej wchodu, uderzył,
Psy ogon w domu słony bitwę, ucisz Radzaną,

Strzeleciała się, szlachta pod polowania się na drogę.
Wreszcie uważał z domu sienkiem swym swym pod kończyło.
Podziękozodano dzielnie pomagnała się twarzono:
Ale Co rzecz z nim do było staremu; przemyśle coraz ma trzy wilkim,

Pod nim za nim wolna wielkim włożył się i pogromadzy.
U to groziem od czas zwyklętą powieści,
Od spoczyła domu polowania na zachodzie od ro
---------------


Kobieta, Brzytwa, KlucznikaJeden wolniej słowy, w nocy słowo;
Do wieku, drugie piorunem przebraniu,
Wieś, że się zuchwały poradości i coraz na niczym,

Zapłotniąc porwały się na zajeżdża okolicy do góry. 
Telimena, Pijaństwo, ŚwiecaPrzysłowili te stodoła w rozkaza. Ojczyzna, Horeszków 
Wiedzi, sposypaniePodkomorzy, Pozycja się co dosyć się kryjomu,
Z szlochatek spałem dwa ogromnemu dosyć w końca.

Przy mymowie z pannym słonie dziwym znakiem krytkiem;
To ziemi wygra do siebie po naszym włos jedn
---------------

Ostrzegam w policy strony
I ten orzechy krzesła okolicy:
Aż jedną przeszła trudną między do trawy

I przecisnął się wydłuż wodą: tuż ona widokej zamku,
Asesor piękne różne, wzdychając boju,
Pomiędzy szlachcic lub w kawych stosach wonie:
Patrzącyc i szlachta szlachtą do rośną bitwy,

Częstości jako widziała tysiąca dawne pannych,
Że podkomorzy graniejszych, jak mowe skroni:
Patrząsł do stołu, do pole słońce pełen wrosną,
Tłumnymi zabiera pokrytkie, trzeje w ręky w położy wysoki,
To przedarty pałk
---------------

Tako kiedy się w progu kłosy się woni,
Ogińskie prawo, trzymawiają na rogi.
Sąd bardzo przyszła niezwierciada: widzą się po koniu

Wysługał na pole szybkim butlona,
Koszyk wciążki i przystrawała się z rzez okna szpica,
Bo niby też na domu od latach spotkania,
Bo potem fortunany ten właśnie senek pała
Z przebuchnianym ogródku do skórę przed kontasznej szkodzie,

Gdy w Niemnej Telimenie, jeszcze dowiedział zdało
O pole tam go z tenczasa na powietrzeć .
Wiem, że po radości nie mieliśmy oczy do nich
---------------

Przesłonęła grozi się na okna wierzgary,
Że w kandarz stawiali się do wita,
W ten na świeży, trzeczność po rynali,

Aż do podskoczył niedozwoli; niedoskali,
Nigdy nieco na niebios — to chciał rodzinę i gromadę,
Że on roliła między świszą na pole ogniste batką,
W drugicie, zwani jedynek nie dworu i cztery poprawiać:
Że wiem do swe wielki przyjaźnie w pośpiewach wielkich myśliwieńców,

Że w miejscach szlachcic krew wojennem chwili się ogromem.
Policman, Podróż, WieczósMaLedwie ten zawił się taką p
---------------

O tymi podaję myślić i panienka,
Tam nagle za gałęziem słową: nie chcą go strony
Wzrok myśląca: Prawda! ot tam wszystkie cały może

Bo tu na plus, co jest zaczęła się się pod gąsienia,
Odwodząc się w zamku przemocę; klęcząc się ponar.
Odejdźmy zamknął się już ciągle zostać;
Widać, że się z obojaśnioną kulę należał. 
Wojski gośćWojski daremniej i pogodził się przed zabawił:

Może widok do zamku jest poszedł swą strony,
W konewkaniu miał po otwarku zamach szamoczyć;
Szaty w domu chwili podpowiedzi
---------------
```

Does this sound like Pan Tadeusz? Not really. But... 
