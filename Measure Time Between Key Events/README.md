![screenshow_mtbk-2.png](https://raw.githubusercontent.com/KarolDuracz/scratchpad/main/Measure%20Time%20Between%20Key%20Events/screenshow_mtbk-2.png)

<b>new_toke.html</b> - Można potem gdy już zrobię ten test 15 tygodniowy "challange" dodać ML jak "makemore" @karpathy, jakiś prosty MLP żeby pomógł w przewidywaniu sekwencji słow. Zamiast klikać na klawiaturze można myszką klikać w kafelki jak na tym obrazku i po prostu w ten sposób tworzyć kod. Nie uderzając w klawisze tylko klikając TOKENY ! Ale to będzie kolejny challange żeby połączyć to z makemore za 15 tygodni ;p
<hr>

02-07-2024 - sposób użycia</br >
1. generujesz sample jak pokazane w sample_generator_v2.ipynb żeby pozbyć się zbętnych komentarzy i pustych linii</br >
2. wklejasz jakiś fragment (u mnie na ten moment nie więcej niż 100 lini kodu linux w 15 min daje radę wklepać około ~3500 znaków wg tego licznika) po prawej</br >
3. naciskasz "analyze sample"</br >
4. naciskasz START - licznik zaczyna mierzyc 15 minut (w milisekundach) - po 15 minutach okno treningowe zostanie ustawione na DISABLED=true</br >
5. naciskasz STOP żeby zapisać log na dysk</br >
6. W ead_stats_from_typerAPP.ipynb masz pokazane jak odczytac jakiś fragment z tego loga</br >
<br />
02-07-2024 - Update - <b>index3.html</b> - Aktualny poprawiony. Dodane zapisanie wszystkiego do jednego pliku . To jest objekt "_glob_arr_fo_obj". I poprawione jeszcze kilka drobnych rzeczy w kodzie względem index2.html
- read_stats_from_typerAPP.ipynb -> służy do generowania statystyk z zpisanego pliku który zrzuca informacje z _glob_arr_fo_obj
- sample_generator_v2.ipynb -> tutaj mam zamiar dodać jeszcze losowe wycinanie kodu z całego folderu repo, ale tutaj jest przykład jak obrobić plik w 2 krokach (stage 1, 2) żeby pozbyć się komentarzy i zbędnych linii, które powodują że aplikacja źle liczy linie. Ponieważ obecnie puste linie nie są brane przez "komparator" między trening vs sample area. Ale sam counter liczy to jako linie jeśli przesuniesz kursorem albo zrobisz enter. Dlatego jest ten guzik "analyze sample" który wcześniej scalał tekst. Teraz robione jest już na tym etapie w pythonie w 2 krokach.
- dodane "alert()" pod przyciskiem "check line sample" . W konsoli też się pojawia ale alert jest szybszy do analizy co tam jest ile razy tab, ile razy spacja
<hr>
29-06-2024 - Update -
<b>index2.html</b> 
- Najpierw wklej kod po prawej stronie, i naciśnij "analyze sample" następnie START. I możesz pisać po lewej. To pierwsze wycina z kodu puste linie, czyli scala kod pozbywając się pustych lini. Start uruchamia resztę skrytpów żeby po wpisaniu znaku po lewej coś się działo na dole.
- poprawilem troche licznik lini - ale jest jeszcze błąd w przypadku cofania backspace wtedy trzeba strzałkami wrócić do linii 0 i znowu w dół do aktualnej linii. Patrz wtedy na div w prawym dolnym rogu
- poprawiona analiza błędów w pisowni linia treningowa vs sample
- dodane kilka licznikow w tym znaki na minutę
- "check line sample" jest po to, żeby w przypadku dziwnych odstępów w kodzie zobaczyć z czego jest zbudowany np "\t\t\t     const struct" czyli 5xtab + 5x' ' - mozna to sprawdzić w konsoli żeby nie świeciło na czerwono że robisz błąd
  ps: to jest aplikacja dla własnego użytku, dla sprawdzenia i poprawy szybkości pisania kodu ogólnie . Wstępna wersja jest ok dla mnie.
<hr>

Do użytku własnego. Aplikacja ma śledzić postępy w szybkości kodowania, wyrabiania nawyków pisania na klawiaturze. Na szybko zrobiłem przez noc, więc nie ma wielu rzeczy i jest błąd gdzies w liniach <br/>
index.html:513 -    Uncaught TypeError: Cannot read properties of undefined (reading 'length') at keyEventWorker.onmessage (index.html:513:40)
</br >
W pliku <b>JS_keytyper_counter_app.ipynb</b> jest przykład statystyk z plików które zapisywane są w /Downloads (2 pliki .txt)
![screenshot_mtbke.png](https://github.com/KarolDuracz/scratchpad/blob/main/Measure%20Time%20Between%20Key%20Events/screenshot_mtbke.png)
