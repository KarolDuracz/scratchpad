#include <windows.h>
#include <stdio.h>
#include <psapi.h>
#include <iostream>

// testowy licznik dla klikniec myszy i klawiatury
// tylko to jest niebezpieczne w pewnych sytuacjach. Bo jeśli będzie robiony przelew moze cos poklikac nie tam gdzie trzeba.
// albo jesli bedzie tam cos operowane a nie ze jest tam pasek adresu.
// zreszta mam 2 monitowy. Wiec trzeba dodac aktualny obszar roboczy czyli gdzie aktualnie jest kursor - lewy czy prawy ekran.
// jesli zmienia sie aktualna karta, czyli to też trzeba śledzić wtedy mozna resetować te liczniki. I gdy jest 0 wykona te funkcje.
volatile int click_counter_down = 0;
volatile int click_counter_up = 0;
volatile int click_counter_key = 0;

// globalna zmienna dla pozycji kursowa zanim dojdzie do pobrania i wysłania inputs
POINT global_cur_pos;

// edytując np plik https://github.com/KarolDuracz/scratchpad/edit/main/ProcessDetector/simple_detector.cpp
// długosc stringa w buf to około 137. Czyli wartość testowa 100 musiała być zmieniona bo pojawił się kolejny błąd
// najpierw miałem błąd przy otwarciu nowego okna z buf size = 0, teraz większy string od bufora.
// dlatego jest globalnie zdefiniowana wartość. Dla google to pewnie nawet 500 to za mało bo niektore zapytania
// sa dosc dlugie ale poki co dla testow niech bedzie 1000 - 25.05.2024 - ustawienia 
#define BUFSIZE 1000

// wątek główny sprawdzający na rejestrze FS:0x30 aktualny PID procesu
// żeby nie blokować całego wątku sprawdzanie co 1 sekunde, tylko robić
// to w petli aktywnie co 1000 ms dla przykładu, a dopiero po zmianie 
// numeru wykonać akcje OpenProcess, Get... etc.
// Innymi słowy lepiej stworzyć wątek dla samego sprawdzania co jest aktualnie
// na pierwszym planie i dopiero po jakieś reakcja typu CALLBACK otwierać handler
// do procesu i coś tam działać. 

// mozna uzyc DEFYKOWANYCH funkcji bo np po co porównywać cały tekst 
// skoro można porównać kilka bajtów i jak się nie zgadza coś to przerwać pętlę
// i zwrócić wynik że zmieniła się nazwa i wykonać dalszą część kodu 
// jeśli już tak bardzo chce optymalizować kod............................

typedef struct {
	char buf[100];
} HT;

struct chr_list {
	int hash;
	std::string str;
};

//typedef struct {
//
//} TAB_NAMES;

static struct chr_list *hashtab[100];

volatile int counter = 0;

// testowy sztywny bufor typu HASH SET do gromadzenia informacji o unikalnych hash
// które zostały wygenerowane na podstawie nazw
// niech bedzie na sztwyno na razie dla testow 500
// w funkcji wykonujacej alokowanie hash dodam info sprawdzajace wielkosc bufora i ilosc dostepnego miejsca
static int hashed_unique_buffer[500];
static int hashed_unique_buffer_idx = 0;

void __alloc_to_hashed_unique_buffer(int hash)
{
	// wyszukiwanie liniowe całego bufora
	for (int i = 0; i < 500; i++) {
		int k = hashed_unique_buffer[i];
		std::cout << hashed_unique_buffer[i] << std::endl;
		if (k == hash) {
			std::cout << " __alloc_to_hashed_unique_buffer --> HASH ZNADUJE SIE W TABLICY " << std::endl;
			break;
		}
		else {
			hashed_unique_buffer[hashed_unique_buffer_idx] = hash;
			hashed_unique_buffer_idx++;
			std::cout << "__alloc_to_hashed_unique_buffer --> zaalokowany nowy hash " << hash << std::endl;
			std::cout << "__alloc_to_hashed_unique_buffer --> aktualny rozmiar tablicy na mozliwysch 500 pozycji to : " << hashed_unique_buffer_idx << std::endl;
			break;
		}
	}
}

// możliwych milion unikalnych kombinacji 
#define HASHSIZE 1000000 

unsigned __hash(char* s)
{
	unsigned hashval;
	for (hashval = 0; *s != 0; s++)
		hashval = *s + 31 * hashval;
	return hashval % HASHSIZE;
}

unsigned __hash2(const char* s)
{
	unsigned hashval;
	for (hashval = 0; *s != 0; s++)
		hashval = *s + 31 * hashval;
	//__alloc_to_hashed_unique_buffer(hashval % HASHSIZE);
	return hashval % HASHSIZE;
}

void __get_local_time()
{
	SYSTEMTIME st;
	GetLocalTime(&st);
	printf("\\\\\\\\\\\\\\\\\\\ %02d | %02d:%02d:%02d - %d:%d:%d \\\\\\\\\\\\\\\\\\\ \n", 
		st.wDayOfWeek, st.wDay, st.wMonth, st.wYear, st.wHour, st.wMinute, st.wSecond);
}

void __open_handler(HWND hWnd)
{
#define BUF_SIZ 500
	char buf_name[BUF_SIZ];
	DWORD pid = 0;
	GetWindowThreadProcessId(hWnd, &pid);
	HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_TERMINATE, FALSE, pid);
	DWORD dump_pid_name = GetProcessImageFileNameA(hProcess, (LPSTR)buf_name, BUF_SIZ);
	CloseHandle(hProcess);
	printf("%d %s \n", dump_pid_name, buf_name);
	// sprowadz nazwy do 32 bitowych liczb tzw hash
	printf("process hash : %d \n", __hash(buf_name));
	
}

void __get_correc_name(char buf[])
{
	// w przypadku YT jest następujący pattern nazwy
	/*
	.current name hash : 44
	2 40 Hz Brain Activation Binaural Beats: Activate 100% of Your Brain, Gamma Waves - YouTube i jeszc
	e C:\Users\kdhome\source\repos\Project4\Debug\Project4.exe
	05 | 24:05:2024 - 13:12:5
	81 \Device\HarddiskVolume2\Program Files (x86)\Microsoft\Edge\Application\msedge.exe
	process hash : 49
	current name hash : 19
	.current name hash : 19
	1 ?? ???????? ź????? ? ???????╗ 22.05 - YouTube i jeszcze 12 stron Ś [InPrivate] Ś Microsoft? Edge
	0 Hz Brain Activation Binaural Beats: Activate 100% of Your Brain, Gamma Waves - YouTube i jeszcze
	05 | 24:05:2024 - 13:12:8
	81 \Device\HarddiskVolume2\Program Files (x86)\Microsoft\Edge\Application\msedge.exe
	process hash : 49
	current name hash : 47
	*/
	// po wykryciu - YouTube trzeba przerwać dalszą iteracje i zwrócić nazwę BEZ "i jeszcze..."
	// dopiero z tego obliczyć hash ponieważ dodawanie kolejnych kart zmienia hash np 12 i 13 otwarych kart
	// generuje inny hash dla tego samego okna msedge
	const char pattern[] = "- YouTube";
	int i, j;
	//static char temp_buf1[50];
	int prev = 0;
	int curr = 0;
	int index_count_prev = 0;
	int index_count_curr = 0;
	for (i = 0; buf[i] != '\0'; i++) {
		//printf("%c ", buf[i]);
		//printf("%d ", lstrlenA(pattern));
		for (j = 0; j < lstrlenA(pattern); j++) {
			if ((buf[i] == pattern[j])) {
				printf("%d %c %c %d \n", i, buf[i], pattern[j], (buf[i] == pattern[j]));
				// sprawdz ciaglosc stringa
				curr = buf[i];
				index_count_curr = i;
				printf(" swap test %c %c %c %d \n", curr, prev, pattern[j], (index_count_curr-index_count_prev));
				prev = curr;
				index_count_prev = index_count_curr;
			}
		}
	}
}

// 24-05-2024
// funkcja pobierajaca string do momentu pojawienia sie paternu "- YouTube"
// to jest potrzebne poniewaz w zależności o ilości otwartych kart w przeglądarce
// generowany jest rózny hash mimo tej samej nazwy aktualnej karty.
// stąd takie podejście.
void __get_correct_name2(char buf[])
{
	const char pattern[] = "- YouTube";
	std::string str(buf, lstrlenA(buf));
	std::cout << "{buf} " << str << " " << (str.find(pattern) & 0xffff) << std::endl;

	// popatrz na koncu tego funkcji na komentarz
	// trzeba by dodać może długość stringa 
	// zanim dojdzie do obliczenia hasha
	// bo strona główna YT też musi być liczona
	// 39 - dla 1 strony pustej YT
	// 58 - dla kilku kart
	// 59 - dla kilunastu kart????
	std::cout << "len of buf " << lstrlenA(buf) << std::endl;
	
	// wyizoluj ciag
	std::string ret_string;
	if ((str.find(pattern) & 0xffff) < 0xffff) {
		ret_string = str.substr(0, str.find(pattern));
	}

	std::cout << "zwracany string " << ret_string << std::endl;

	// check hash
	std::cout << "hash = " << __hash2(ret_string.c_str()) << std::endl;

	// ale dla pewnosci warto po prostu porownas string z tym co jest w bazie danych
	// coś ala install ze str 85 (168) ANSIC
	//struct chr_list* np;
	////np = hashtab[__hash2(ret_string.c_str())];
	//std::cout << " np = " << hashtab[__hash2(ret_string.c_str())] << std::endl;
	//if (hashtab[__hash2(ret_string.c_str())] == NULL) {
	//	hashtab[__hash2(ret_string.c_str())] = (struct chr_list*)malloc(sizeof(chr_list));
	//	/*if (__hash2(ret_string.c_str()) != 0) {
	//		np = hashtab[__hash2(ret_string.c_str())];
	//		np->hash = __hash2(ret_string.c_str());
	//		np->str = ret_string.c_str();
	//	}*/
	//}
	//else {
	//	std::cout << " hash " << __hash2(ret_string.c_str()) << " is allocate inside hashtable" << std::endl;
	//}
	struct chr_list* np;

	np = hashtab[__hash2(ret_string.c_str())];

	std::cout << " np = " << np << " " << __hash2(ret_string.c_str()) << " " << std::endl;

	// to dziala tylko dla YT
	// i źle że działa, bo pusta strona https://www.youtube.com/
	// zwraca 0 mimo że to YT, dlatego że obcina do tego napisu w string, patternu
	// czyli trzeba zmienić maske na wieksze od 0 albo coś takiego.
	// czyli sprawdzać czy jest choć 1 znak przez "- YouTube"
}

// może prościej będzie WYCIĄĆ ten fragment gdzie jest ilość otwartych kart
// czyli znaleźć ten pattern. I tworzyć hash dla całości bez tych cyferek
// albo nawet dodawać zamiast tego 3x0 czyli 000 w miejsce tych cyferek
// wtedy dostaję pattern dla opcji z wieloma kartami (a nie będzie raczej otwartych ponad 1000 kart)
// i wtedy są 2 opcje dla strony głównej i dla otwarcyh kart.
void __get_correct_name3(char buf[])
{

	// wersja dla 1 pustej karty w trybie private
	// TODO
	// 
	// 
	//------------------------------------------------------------------
	// TODO - to dziala dla strony glownej YT z kilkoma kartami otwartymi obok
	std::string str(buf, lstrlenA(buf));
	std::cout << " entry point name3 : " << str << std::endl;
	std::cout << str.find("jeszcze") << " " << str.find("strony") << " " << lstrlenA("jeszcze") << std::endl;

	// jesli jest pusta 1 karta w trybie private to wyżej zwróci wartości które są out of range = program się skraszuje
	// << COS TO NIE DZIALA JAK TRZEAB --------- TODO
	// jak wejdziesz na nudecollect to zobaczysz ze wykonuje "return to main loop" mimo ze to nie jest ta sama nazwa co pattern ;/
	const char pattern_2[] = "Nowa karta InPrivate Ś [InPrivate] Ś Microsoft? Edge";
	std::cout << "cmp " << strcmp(str.c_str(), pattern_2) << std::endl;
	int test_1 = strcmp(str.c_str(), pattern_2);
	std::cout << " test val = " << test_1 << std::endl;
	if (test_1 == 1) { 
		// na razie po prostu wyjdę z funkcji tutaj żeby nie było błędu
		// i tak wraca do pętli głównej
		std::cout << " return to main loop " << std::endl;
		return;
	}

	//size_t index = 10 + 7 + 1; // to jest tylko dla YT
	size_t index = str.find("jeszcze") + 7 + 1; // to dziala na kazdej stronie z otwartymi kartami obok
	// sa jeszcze bledy wynikajace z braku czyszczenia bufora np przy edycji na githubie wywali blad tutaj tez
	std::cout << "VERRY IMPORTANT TEST - index value : " << index << " " << lstrlenA(buf) << " " << buf << std::endl;
	// dodanie protekcji żeby nie wychodziło poza OUT OF RANGE
	if ((index + 1) >= lstrlenA(buf) /* || lstrlenA(buf) > index*/) {
		std::cout << "[ try to get access to out of range index -> return from this function right now." << std::endl;
		std::cout << (index + 1) << " "  << lstrlenA(buf)  << " " << ((index + 1) >= lstrlenA(buf)) << " " << (lstrlenA(buf) > index) << std::endl;
		return;
	}
	std::cout << str.at(index) << " " << isdigit(str.at(index)) << " " << isdigit(str.at(index+1)) << std::endl;
	// index + 0 - 0-9
	// index + 1 - 10-99
	// index + 2 - 100-999
	// w sumie nie trzeba sprawdzac czy kolejne indeksy to też digit tylko zrobić offset od razu
	static char new_temp_buf[500];
	if (isdigit(str.at(index)) == 4) {
		//str.substr(str.at(index))
		std::string __first = str.substr(0, index);
		std::string __last = str.substr(index + 2, lstrlenA(buf) + 2);
		std::cout << " first / last " << __first << " || " << __last << std::endl;
		
		// add to new temp buf - first
		int i;
		for (i = 0; i < lstrlenA(__first.c_str()); i++) {
			new_temp_buf[i] = __first.c_str()[i];
		}

		new_temp_buf[i++] = '|';
		new_temp_buf[i++] = '|';
		new_temp_buf[i++] = '|';
		new_temp_buf[i++] = ' ';

		// add to temp buf - last part
		int j = 0;
		while (j < lstrlenA(__last.c_str())) {
			new_temp_buf[i] = __last.c_str()[j];
			i++;
			j++;
		}

		std::cout << " new buf after add " << new_temp_buf << std::endl;

	}

}

void __get_cursor_pos()
{
	POINT point;
	GetCursorPos(&point);
	std::cout << " :: cursor position :: " << point.x << " " << point.y << std::endl;

	// mając informacje o ostatnim input który został zmierzony można tylko nasłuchiwać tego zdarzenia
	// i dopiero potem reagować, zamiast co 1000 ms pętlę przetwarzać.
	// z drugiej strony to dziala dla klawiatury i myszy wiec byloby duzo wywołań
	// można spróbować połączyć to z wykrywaniem który proces jest aktualnie na pierzym planie
	LASTINPUTINFO last_inp;
	last_inp.cbSize = sizeof(LASTINPUTINFO);
	GetLastInputInfo(&last_inp);
	std::cout << " :: input time  :: " << last_inp.dwTime << std::endl;

	/*BYTE lpKeyState2[0xff];
	if (GetKeyboardState(lpKeyState2)) {
		std::cout << " :: key state  :: " << lpKeyState2 << std::endl;
	}*/

	/*
	//chat gpt example
	bool IsKeyPressed(int virtualKey)
{
	// Create a 256-byte array to hold the state of each key
	BYTE keyState[256];

	// Get the current state of all keys
	if (GetKeyboardState(keyState))
	{
		// Check if the high-order bit of the key's state is set
		return (keyState[virtualKey] & 0x80) != 0;
	}

	// Return false if the function fails
	return false;
}

int main()
{
	// Virtual key code for the "A" key
	int virtualKeyA = 0x41;

	std::cout << "Press the 'A' key to see if it is detected. Press 'Esc' to exit." << std::endl;

	while (true)
	{
		// Check if the 'A' key is pressed
		if (IsKeyPressed(virtualKeyA))
		{
			std::cout << "'A' key is pressed." << std::endl;
		}

		// Check if the 'Esc' key is pressed to exit the loop
		if (IsKeyPressed(VK_ESCAPE))
		{
			std::cout << "Exiting..." << std::endl;
			break;
		}

		// Sleep for a short duration to avoid spamming the output
		Sleep(100);
	}

	return 0;
}


	*/

}

void __send_input_mouse_down()
{

	if (click_counter_down > 0)
		return;

	/*POINT point;
	GetCursorPos(&point);
	std::cout << " :: cursor position :: " << point.x << " " << point.y << std::endl;*/

	GetCursorPos(&global_cur_pos);

	int x = 2206;
	int y = 50;

	/*INPUT inputs[2] = {};
	ZeroMemory(inputs, sizeof(inputs));

	inputs[0].type = INPUT_MOUSE;
	inputs[0].mi.dwFlags = MOUSEEVENTF_RIGHTDOWN;
	inputs[0].mi.dx = x;
	inputs[0].mi.dy = y;
	inputs[0].mi.mouseData = XBUTTON2;
	inputs[0].mi.dwExtraInfo = 0;

	inputs[1].type = INPUT_MOUSE;
	inputs[1].mi.dwFlags = MOUSEEVENTF_RIGHTUP;
	inputs[1].mi.dx = x;
	inputs[1].mi.dy = y;
	inputs[1].mi.mouseData = XBUTTON2;
	inputs[1].mi.dwExtraInfo = 0;*/

	SetCursorPos(x, y);

	INPUT inputs[1] = {};

	// Set up the right mouse button down event
	inputs[0].type = INPUT_MOUSE;
	inputs[0].mi.dwFlags = MOUSEEVENTF_LEFTDOWN;

	// Set up the right mouse button up event
	/*inputs[1].type = INPUT_MOUSE;
	inputs[1].mi.dwFlags = MOUSEEVENTF_LEFTUP;*/

	// Send the mouse events
	SendInput(1, inputs, sizeof(INPUT));

	click_counter_down++;
}

void __send_input_mouse_up()
{

	if (click_counter_up > 0)
		return;

	/*POINT point;
	GetCursorPos(&point);
	std::cout << " :: cursor position :: " << point.x << " " << point.y << std::endl;*/

	int x = 2206;
	int y = 50;

	/*INPUT inputs[2] = {};
	ZeroMemory(inputs, sizeof(inputs));

	inputs[0].type = INPUT_MOUSE;
	inputs[0].mi.dwFlags = MOUSEEVENTF_RIGHTDOWN;
	inputs[0].mi.dx = x;
	inputs[0].mi.dy = y;
	inputs[0].mi.mouseData = XBUTTON2;
	inputs[0].mi.dwExtraInfo = 0;

	inputs[1].type = INPUT_MOUSE;
	inputs[1].mi.dwFlags = MOUSEEVENTF_RIGHTUP;
	inputs[1].mi.dx = x;
	inputs[1].mi.dy = y;
	inputs[1].mi.mouseData = XBUTTON2;
	inputs[1].mi.dwExtraInfo = 0;*/

	SetCursorPos(x, y);

	INPUT inputs[1] = {};

	// Set up the right mouse button down event
	inputs[0].type = INPUT_MOUSE;
	inputs[0].mi.dwFlags = MOUSEEVENTF_LEFTUP;

	// Set up the right mouse button up event
	/*inputs[1].type = INPUT_MOUSE;
	inputs[1].mi.dwFlags = MOUSEEVENTF_LEFTUP;*/

	// Send the mouse events
	SendInput(1, inputs, sizeof(INPUT));

	click_counter_up++;

	// przywróć poprzednią pozycję kursowa !!!!!!!!!!!!!!!!!!!!!!!!!!
	SetCursorPos(global_cur_pos.x, global_cur_pos.y);
	std::cout << ">>> link skopiowany do schoka - pozycja kursowa przywrocona <<< " << std::endl;

}


void __send_input_test()
{

	if (click_counter_key > 0)
		return;

	//OutputString(L"Sending 'Win-D'\r\n");
	INPUT inputs[4] = {};
	ZeroMemory(inputs, sizeof(inputs));

	inputs[0].type = INPUT_KEYBOARD;
	inputs[0].ki.wVk = VK_CONTROL;

	inputs[1].type = INPUT_KEYBOARD;
	inputs[1].ki.wVk = 'C';

	inputs[2].type = INPUT_KEYBOARD;
	inputs[2].ki.wVk = 'C';
	inputs[2].ki.dwFlags = KEYEVENTF_KEYUP;

	inputs[3].type = INPUT_KEYBOARD;
	inputs[3].ki.wVk = VK_CONTROL;
	inputs[3].ki.dwFlags = KEYEVENTF_KEYUP;

	UINT uSent = SendInput(ARRAYSIZE(inputs), inputs, sizeof(INPUT));
	if (uSent != ARRAYSIZE(inputs))
	{
		//OutputString(L"SendInput failed: 0x%x\n", HRESULT_FROM_WIN32(GetLastError()));
	}

	click_counter_key++;

}

void __get_curent_active_monitor(HWND hwnd)
{

	// first screen dimension 1600 x 900
	// mierzę czas przebywania na obszarze, czyli na którym monitorze jest kursor
	// i na tej podstawie określam czy mam na prawej kliknąć czy po lewej
	// tylko co z rozmiarem okna... 

	/*POINT point;
	GetCursorPos(&point);
	
	LASTINPUTINFO last_inp;
	last_inp.cbSize = sizeof(LASTINPUTINFO);
	GetLastInputInfo(&last_inp);*/

	RECT rect;
	GetWindowRect(hwnd, &rect);

	std::cout << " get active monitor info " << rect.bottom << " " << rect.top << " " << rect.left << " " << rect.right << std::endl;

	
}

int main()
{

	while (1) {

		std::cout << "---------------------------------------------------------------------------------" << std::endl;

		HWND hwnd = GetForegroundWindow();

		//printf("%d \n", hwnd);

		DWORD out;

		int ret;
		ret = GetWindowThreadProcessId(hwnd, &out);

		//printf("[ id ] %d \n", out);

		static int tid;

		// 
		static int x1;
		static int x2;
		static int x3;

#if 0
		__asm 
		{
			//mov eax, fs:0x24
			//int 3
			lea eax, tid
			push ebx
			mov ebx, fs:0x24
			mov [eax], ebx
			pop ebx
		}
#endif

#if 0
		__asm {
			push ebx
			lea ebx, x1
			mov [ebx], eax
			lea ebx, x2
			mov [ebx], ecx
			lea ebx, x3
			mov [ebx], esi 
			pop ebx
		}
#endif

		static char buf[BUFSIZE];
		static char _cur_buf[BUFSIZE];
		// clear buffers
		ZeroMemory(buf, BUFSIZE);
		ZeroMemory(_cur_buf, BUFSIZE);
		// get window title to buffer
		GetWindowTextA(hwnd, buf, BUFSIZE);
		//printf("[1] %s %d %d %d %s \n", buf, x1, x2, x3, _cur_buf);
		printf("current name hash : %d \n", __hash(_cur_buf));

		// musi byc sprawdzanie czy nie jest zero poniewaz przy otwarciu nowego okna z dolnego paska
		// jest moment że nie ma w buforze żadnego stringa i przez to potem w kolejnej funkcji wywala program na sprawdzaniu indexu "at"
		if (lstrlenA(buf) > 0) {
			//__get_correc_name(buf);
			//__get_correct_name2(buf);
			__get_correct_name3(buf);
		}
		else {
			// w sumie to mozna nawet sygnalizować taką operacje, ale to nie jest konieczne chyba w logach.
			// ale jest cos co powoduje ze po otwarciu okna, potem zamknieci i nie kliknieci gdzies indziej 
			// ten log sie bedzie pojawial dalej wiec lepiej go do LOGOW nie zapisywac, tylko jako debug programu
			// teraz w trybie testow niech leci sobie te w konsoli
			std::cout << "[ otwarcie nowego okna ]" << std::endl;
		}

		int cmp_ret = strcmp(buf, _cur_buf);
		//printf("string cmp %d \n", cmp_ret);

		// jesli nie jest rowne zero -1 / 1 wtedy zrob funkcje na lockach i bardziej wymagajace 
		if (cmp_ret != 0) {
			printf("%d %s %s\n", counter, buf, _cur_buf);
			__get_local_time();
			__open_handler(hwnd);
			__get_curent_active_monitor(hwnd);
			__get_cursor_pos();
			__send_input_mouse_down();
			__send_input_test();
			__send_input_mouse_up();
			counter = 0;
		}
		else {
			printf(".");
			counter++;
		}

		strcpy_s(_cur_buf, BUFSIZE, buf);

		//printf("[         2] %s %d %d %d %s \n", buf, x1, x2, x3, _cur_buf);

		Sleep(1000);

	}

	return 0;
}

/*
int main()
{

	HDC screen = GetDC(NULL);
	HDC screen2 = GetDC(NULL);
	HBRUSH red = CreateSolidBrush(RGB(255, 0, 0));
	HBRUSH blue = CreateSolidBrush(RGB(255, 255, 0));
	SelectObject(screen, red);
	SelectObject(screen2, blue);

	int step = 0;

	int x_off = 1200;

	// text
	static char buf[1000];
	// get some chars
	int c;
	int k = 0;
	while ((c = getchar()) != EOF) {
		buf[k] = c;
		k+=2;
	}
	buf[k] = '\0';
	printf("%ws %d\n", buf, k);
	// ...
	int len = 0;
	for (int i = 0; buf[i] != '\0'; i+=2) {
		len = i;
	}
	TCHAR* text = (TCHAR*)malloc(len);
	TextOut(screen, 10, 50, (TCHAR*)buf, len);
	printf("%d %p\n", len, text);
	while (1)
	{
		Rectangle(screen, 50 + x_off, 50, 500, 500);
		//Rectangle(screen, 100 + x_off, 150, 500, 400 + step);
		
		Sleep(1);
		step += 1;
	}
	return 0;
}
*/
