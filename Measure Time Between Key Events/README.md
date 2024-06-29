29-06-2024 - Update -
użyj <b>index2.html</b> 
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
