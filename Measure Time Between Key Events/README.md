29-06-2024 - Update -
użyj <b>index2.html</b> 
- poprawilem troche licznik lini - ale jest jeszcze błąd w przypadku cofania backspace wtedy trzeba strzałkami wrócić do linii 0 i znowu w dół do aktualnej linii. Patrz wtedy na div w prawym dolnym rogu
- poprawiona analiza błędów w pisowni linia treningowa vs sample
- dodane kilka licznikow w tym znaki na minutę
  ps: to jest aplikacja dla własnego użytku, dla sprawdzenia i poprawy szybkości pisania kodu ogólnie . Wstępna wersja jest ok dla mnie.
<hr>

Do użytku własnego. Aplikacja ma śledzić postępy w szybkości kodowania, wyrabiania nawyków pisania na klawiaturze. Na szybko zrobiłem przez noc, więc nie ma wielu rzeczy i jest błąd gdzies w liniach <br/>
index.html:513 -    Uncaught TypeError: Cannot read properties of undefined (reading 'length') at keyEventWorker.onmessage (index.html:513:40)
</br >
W pliku <b>JS_keytyper_counter_app.ipynb</b> jest przykład statystyk z plików które zapisywane są w /Downloads (2 pliki .txt)
![screenshot_mtbke.png](https://github.com/KarolDuracz/scratchpad/blob/main/Measure%20Time%20Between%20Key%20Events/screenshot_mtbke.png)
