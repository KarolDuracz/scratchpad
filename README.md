# scratchpad
update : 05.09.2024 : This "scratchpad" repo is theoretically closed. I won't be posting anything more here. These are just a few topics that interest me. Probably end of 2025 I'll be back to posting if there's something interesting finally worth sharing and developing. TIME TO GO TO NEXT LEVEL ...in programming skills also. 
<br /><br />
NtQueryInformationProcess_run_example - dump.png
![dump](https://raw.githubusercontent.com/KarolDuracz/scratchpad/main/dump.png)

<b>64 bit - dla 32 bitowego procesu offsety są inne i nie wczyta adresu PEB a co z tym idzie reszty odczytywanej przez ReadProcessMemory.</b>

<i>Dodaj ten kod po lini 85, czyli po printf("%zu \n", sizeof(PVOID));</i><br>
Te struktury i pola bitowe jak IMAGE_DOS_SIGNATURE są w winnt.h
```
IMAGE_DOS_HEADER *idh = (IMAGE_DOS_HEADER*)hdr;			
printf("e_magic: %p %p %d %d %p\n", idh, image, IMAGE_DOS_SIGNATURE, idh->e_magic, idh->e_lfanew);

IMAGE_NT_HEADERS32 *inh32 = (IMAGE_NT_HEADERS32*)((BYTE*)hdr + idh->e_lfanew);
printf("%d %d \n", inh32->Signature, IMAGE_NT_SIGNATURE);
```
Zwróci coś takiego:

e_magic: 000000B6B3A0E0F0 00007FF6B9400000 23117 23117 00000000000000F8<br />
17744 17744
<br/>
Czyli pobrane pola są takie same jak IMAGE_DOS_SIGNATURE i IMAGE_NT_SIGNATURE. W ten sposób można dostawać się do pól tych struktur.
http://pinvoke.net/default.aspx/Structures.IMAGE_DOS_HEADER
Ten offset 0xf8 z idh->e_lfanew jest ten sam który obliczyłem "ręcznie" tylko mi wyszło 247.
<br /><br />
https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-image_optional_header32
```
IMAGE_NT_HEADERS32 *inh32 = (IMAGE_NT_HEADERS32*)((BYTE*)hdr + idh->e_lfanew);
printf("%d %d \n", inh32->Signature, IMAGE_NT_SIGNATURE);
printf("%p %d %p %d\n", inh32->OptionalHeader.AddressOfEntryPoint,
inh32->OptionalHeader.SizeOfImage, inh32->OptionalHeader.ImageBase, inh32->OptionalHeader.SizeOfCode);
```
0000000000011276 159744 0000000000007FF7 37376
<hr>

![ntqueryinfo_example2.png](https://raw.githubusercontent.com/KarolDuracz/scratchpad/main/ntqueryinfo_example2.png)

update 23-07-2023 - 
<b>NtQueryInformationProcess_run_example</b>
<br />
Linie 145-147 - odczytanie linii poleceń. To jest przydatne gdy aplikacja jest wywoływana z argumentami, a tak jest np przy otwarciu pliku w notatniku (nie pustego notatnika, tylko pliku .txt). Wtedy w lini poleceń jest ścieżka do pliku. Tak samo proces msedge.exe, czyli przeglądarka internetowa, w lini poleceń jest sporo informacji do wyciągnięcia. <br />

```
ReadProcessMemory(h, rupp.CommandLine.Buffer, cmd, rupp.CommandLine.Length - 2, &rb4);
printf("%d %ws\n", rb4, cmd);
```

<hr>

![EnumWindows_img.png](https://raw.githubusercontent.com/KarolDuracz/scratchpad/main/EnumWindows_img-a.png)

Jeśli chciałbym znaleźć aktualnie otwartą kartę w przeglądarce, aktualnie aktywne okno i to co widnieje w "title" to jak to zrobić? Okazuje się że EnumWindows w tej formie wyciągnie te informacje, ale potrzeba moim zdaniem jest jeszcze kolejna funkcja PrintProcessNameAndID która zwraca ściżkę do procesu (ścieżkę uruchomionego programu np. msedge.exe) oraz PID. Dzięki temu można w dalszym kroku powiązać ze sobą te dwa wyniki zwracane przez funkcje i filtrować (wyszukiwać) konkretnego PID, w tym przypadku dla procesu "msedge.exe", jeśi chcę znać aktualną nawzę otwartego okna. To jest tylko mały przykład od czego zacząć.

![tasklist verbose.png](https://raw.githubusercontent.com/KarolDuracz/scratchpad/main/taskslit%20verbose-a.png)

Gdyby użyć polecenia CMD <b>tasklist /v</b> to nie dostaniemy wystarczajaco informacji o otwartym oknie. Niektóre otwarte strony nawet nie zwrócą tutaj żadnyuch informacji tylko zobaczymy N/A przy wywołaniu tego polecenia. Dlatego trzeba użyć funkcji z winapi żeby dostać się chociaż do szczątkowych informacji o tym jaka jest aktualna nazwa okna / wyswietlana strona w przeglądarce.

<hr>

![tasklist verbose.png](https://raw.githubusercontent.com/KarolDuracz/scratchpad/main/another_way_to_track_windows.png)

Może to jest najlepsza opcja żeby śledzić otwarte okna w systemie Windows. Jest najbardziej oczywista. Oprócz nazwy okna, PID procesu w drugiej lini jest nazwa procesu, żeby miec pewność który dokładnie proces był w danym momencie był zapisywany w logach / które okno było aktualnie na pierwszym planie wyświetlane. Czyli dostajemy w ten sposób nazwy otwartych stron (nie adresy, ale nazwy z title) oraz nazwę i PID procesu.

```
#define _CRT_SECURE_NO_WARNINGS
#include <string>
#include <iostream>
#include <windows.h>
#include <stdio.h>
#include <psapi.h>
#include <tchar.h>

#define BUF_SIZ 500

FILE* stream_log;

void save_to_file(SYSTEMTIME lt, char *retbuf, int counter)
{
	// logs 
	char log_buf[140] = { 0 };
	char log_tmp_buf[100] = { 0 };
	strcat(log_buf, "c:\\__bin\\detect_ext_logs\\v2-");

	memset(log_tmp_buf, 0, 100);
	_itoa(lt.wYear, log_tmp_buf, 10);
	strcat(log_buf, log_tmp_buf);
	strcat(log_buf, "-");

	memset(log_tmp_buf, 0, 100);
	_itoa(lt.wMonth, log_tmp_buf, 10);
	strcat(log_buf, log_tmp_buf);
	strcat(log_buf, "-");

	memset(log_tmp_buf, 0, 100);
	_itoa(lt.wDay, log_tmp_buf, 10);
	strcat(log_buf, log_tmp_buf);


	memset(log_tmp_buf, 0, 100);
	strcat(log_buf, ".txt");

	//fopen_s(&stream, log_buf, "a");

	printf("%s \n", log_buf);

	fopen_s(&stream_log, log_buf, "a");
	fprintf(stream_log, "%d - %02d | %02d:%02d:%02d ", counter, lt.wDayOfWeek, lt.wDay, lt.wMonth, lt.wYear);
	fprintf(stream_log, "%02d:%02d:%02d", lt.wHour, lt.wMinute, lt.wSecond);
	//fprintf(stream_log, "%s\n", " --logs only.");
	fprintf(stream_log, " %s\n", retbuf);
	fclose(stream_log);
}

int main()
{

	//HANDLE hProcess = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, 9200);

	volatile int i = 0;

	// temporary buffer for first get window name
	char* buffer = new char[BUF_SIZ];

	// the same size - 500
	char* retbuf = new char[BUF_SIZ];

	char buf_name[BUF_SIZ];

	SYSTEMTIME st;

	while (1) {

		HWND hWnd = GetForegroundWindow();

		GetWindowTextW(hWnd, (LPWSTR)buffer, BUF_SIZ);

		int len = GetWindowTextLengthA(hWnd);

		//printf("size of buffer %d %d\n", sizeof(buffer), len);
		
		int j;
		int jj = 0;
		for (j = 0; j < len*2; j++) {
			if (buffer[j] == 0) continue;
			//printf("%c", buffer[j]);
			retbuf[jj] = buffer[j];
			jj++;
		}
		retbuf[jj++] = '\0';

		GetLocalTime(&st);

		DWORD pid = 0;

		GetWindowThreadProcessId(hWnd, &pid);

		printf("%d [%d] - %d:%d:%d - %s\n", i, pid, st.wHour, st.wMinute, st.wSecond,  retbuf);

		HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, pid);

		DWORD dump_pid_name = GetProcessImageFileNameA(hProcess, (LPSTR)buf_name, BUF_SIZ);

		printf("%d %s \n", dump_pid_name, buf_name);

		//save_to_file(st, retbuf, i);

		//printf("%d %ws \n", i, buffer);

		// inc logs counter
		i++;

		Sleep(1000);
	}

	return 0;
}
```
