<b>follow this instructions</b> </br>
press -> WinKey(key) + R </br>
type -> shell:startup </br>
This opens the location -> C:\Users\\{username}\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup </br>
create and copy shortcut of this .exe file to this location  </br>
press -> key + R </br>
type -> msconfig </br>
In the tab "Startup" click "Open Task Manager" and check if the application is on the list </br>
Your app is run at system startup </br>

 </br></br>
<b>[PL] Krótki opis programu </b></br>
Kompilowane MS Studio 2022. To wersja testowa robiona dla własnych celów. Ma śledzić konkretne procesy po nazwie okna i zamykać je, jeśli wykryje któryś z listy procesów do zamknięcia. Lista w tym przykładzie to `std::vector<std::string> patter_list;` do której są dodawane nazwy które mają być śledzone przez aplikację np linia 89 w kodzie `patter_list.push_back("xtensions");`, która śledzi czy zostało otwrte okno msedge Extensions. (Można dodać opcję z odczytywaniem listy z pliku .txt ale to było robione do moich konretnych zastosowań więc dodawanie w ten sposób mi wystarcza)</br></br>
W lini 19 trzeba zmienić ścieżkę do folderu logów `strcat(log_buf, "c:\\__bin\\detect_ext_logs\\v2-all");`. To tworzy plik v2-all.txt ponieważ w lini 38 jest dodany typ pliku `strcat(log_buf, ".txt");`. To znaczy że zapisuje wszystko tylko do jednego pliku. Dodaje na końcu, opcja "a", linia 45 fopen_s.</br></br>
Żeby zobaczyć co dokładnie jest na wyjściu zmień linie 83 `SetWindowPos(hWnd, NULL, 400, 400, 100, 100, SWP_HIDEWINDOW);` która ukrywa okno CMD z paska. Aplikacja działa w tle. Można podejrzeć to np ProcessExplorer. Następnie zakomentuj linie 181 która wywołuje funkcję `my_system_shutdown();`, która wylogowuje z sesji. Następnie linie 161 i 162 `TerminateProcess(hProcess, NULL);`, które zapisują logi do pliku na dysku, a TerminateProcess zamyka proces w systemie. To spowoduje że okno CMD nie zniknie, tylko będzie można prześledzić co 1 sekundę jakie okno jest w tej chwili na pierwszym planie i jego nazwę.

<br /><br/>
TODO<br />
Aplikacja jednowątkowa - minusy - Funkcja MessageBox blokuje wątek. Do póki okno nie zniknie wątek nie jest w stanie wykonywać obliczenia czasu który wyświetla okno co 15 minut w tym przypadku. Dlatego trzeba podzielić program na funkcje które zajmują się np obliczaniem czasu, jak timery, innny wątek wyświetlaniem okienek itd. Wtedy gdy okno jest wyświetlone inny watek mimo to będzie pracować i liczyć delte czasu - &#916;t

<hr>

![xample_process_detector2.png](https://raw.githubusercontent.com/KarolDuracz/scratchpad/main/ProcessDetector/example_process_detector2.png)
<br />
Warto pomyśleć żeby śledzić bardziej ostatni czas zdarzenia czyli ostatnie 3 linie w logu na tym obrazku. Ale jest kilka kwestii jeszcze do ogarnięcia (TODO) bo są pewne okna czasowe i nie można przesadzać z zapisywanie informacji oraz trzeba uważać gdzie myszka klika. Ale na ten moment 27-05-2024 to tyle.
<br />
Ogólnie jest to tylko wstęp, bo miałem chwilę to coś próbowałem rozkmianiać dalej. Wszystko jest pomieszane ale można wybrać z tego kilka rzeczy. Przede wszystkim żeby to zaczęło działać trzeba wybrać który bufor ma być brany pod uwagę przy obliczaniu hash. Następnie trzeba gdzieś dodawać informacje o tych unikalnych id. Do tego miał służyć hashtab, który jest tablicą na wskaźniki, ale równie dobrze może to być bufor o rozmiarze 500-1000 typu int, na właśnie te hash unikalne, i można to liniowo wyszukiwać i dodawać. Albo zrobić jak w ansic lookup table tylko przy rozmiarze 1.000.000 nie ma sensu robić czegoś takiego w tym wypadku, tylko lepiej po prostu liniowo indeks po indeksie kolejne sprawdzać czy hash już jest i inkrementować index. Czyli dodawać nowe na kolejnym indeksie. Ale na ten moment wstępnie tyle.
