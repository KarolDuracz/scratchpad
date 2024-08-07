#define _CRT_SECURE_NO_WARNINGS

#include <Windows.h>
#include <stdio.h>
#include <tchar.h>

FILE* stream_log;
volatile int counter = 0;
volatile int sig_th1 = 0;
volatile int rinse_alert = 0;
volatile int the_signal_came = 0;
volatile int rise_error_signal = 0; // some error undefined behaviour
volatile int rise_error_counter = 0;
HWND hwnd_start = NULL;

#define MAX_THREADS 3
DWORD th_ids[MAX_THREADS];
HANDLE  hThreadArray[MAX_THREADS];

void save_to_file(SYSTEMTIME lt, char *retbuf)
{
	// logs 
	char log_buf[140] = { 0 };
	char log_tmp_buf[100] = { 0 };
	//strcat(log_buf, "C:\\Windows\\Temp\\v7-all");
	strcat(log_buf, "C:\\Users\\{path to file}");
	strcat(log_buf, ".txt");
	printf("%s \n", log_buf);
	fopen_s(&stream_log, log_buf, "a");
	fprintf(stream_log, "%d - %02d-%02d-%02d ",  lt.wDayOfWeek, lt.wDay, lt.wMonth, lt.wYear);
	fprintf(stream_log, "%02d:%02d:%02d", lt.wHour, lt.wMinute, lt.wSecond);
	fprintf(stream_log, " | %s\n", retbuf);
	fclose(stream_log);
}

int __is_visible()
{
	if (hwnd_start != NULL) {
		//if (IsWindowVisible(hwnd_start)) {
		if (IsIconic(hwnd_start)) {
			return 1;
		}
		else {
			return 2;
		}
	}
	return 0;
}

// https://learn.microsoft.com/en-us/windows/console/clearing-the-screen
void cls(HANDLE hConsole)
{
	COORD coordScreen = { 0, 0 };    // home for the cursor
	DWORD cCharsWritten;
	CONSOLE_SCREEN_BUFFER_INFO csbi;
	DWORD dwConSize;

	// Get the number of character cells in the current buffer.
	if (!GetConsoleScreenBufferInfo(hConsole, &csbi))
	{
		return;
	}

	dwConSize = csbi.dwSize.X * csbi.dwSize.Y;

	// Fill the entire screen with blanks.
	if (!FillConsoleOutputCharacter(hConsole,        // Handle to console screen buffer
		(TCHAR)' ',      // Character to write to the buffer
		dwConSize,       // Number of cells to write
		coordScreen,     // Coordinates of first cell
		&cCharsWritten)) // Receive number of characters written
	{
		return;
	}

	// Get the current text attribute.
	if (!GetConsoleScreenBufferInfo(hConsole, &csbi))
	{
		return;
	}

	// Set the buffer's attributes accordingly.
	if (!FillConsoleOutputAttribute(hConsole,         // Handle to console screen buffer
		csbi.wAttributes, // Character attributes to use
		dwConSize,        // Number of cells to set attribute
		coordScreen,      // Coordinates of first cell
		&cCharsWritten))  // Receive number of characters written
	{
		return;
	}

	// Put the cursor at its home coordinates.
	SetConsoleCursorPosition(hConsole, coordScreen);
}

/*
	co 5 minut powinno sie pokazac okno decyzji
	ale LOOK_AHEAD_TIME blokuje watek ktory wyswietla komunikaty
	jesli czas nie jest wiekszy lub rowny tej stalej
	troche przekombinowane ale to jest po to zeby nie wlaczac o 9:00, 10:00 
	tylko o dowolnej porze i samo bedzie szukalo tych minut zeby odpalic alert
	o to tylko chodzi, zeby nie czekac na rowna 8, 9, 10 etc
	-------------------
	Czyli program działa metodą PULL-UP w aktywnej pętli i sprawdza cyklicznie stan dajac opcje wyboru dla uzytkownika 1-n od minuty LOOK_AHEAD_TIME
	ale uzytkownik moze pokazac okno i wtedy wpisac cos sam - !!! - powinien wpisac 4 zeby bylo jasne o co chodzi w logach
*/
// domyslny czas co jaki ma pojawiac sie okno wyboru
#define TIME_SETUP (60 * 5)
// od ktorej minuty mam sprawdzac i zapisywac log
#define LOOK_AHEAD_TIME 45

DWORD WINAPI func_th1(LPVOID lpParam)
{
	while (1) {
		//if (counter >= (TIME_SETUP) && sig_th1 == 1 && rinse_alert == 1) {
		if (sig_th1 == 1 && rinse_alert == 1) {

			HANDLE hStdout;
			hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
			cls(hStdout);

			// show window
			ShowWindow(hwnd_start, SW_RESTORE);

			printf(" ==== LEGEND and some text here ==== \n");
			printf("[ 1 ] -> mode 1 \n");
			printf("[ 2 ] -> mode 2 \n");
			printf("[ 3 ] -> mode 3 \n");
			printf("[ 4 ] -> Async event : mode 4.\n");
			printf("[ 5 ] -> PRINT HELP and do nothing, don't save to file this event.\n");
			printf("-------------------------------------------\n");

			SYSTEMTIME st;

			int c;
			char buf[2]; // 2 poniewaz potrzeba miejsce na liczbe 1-n + znak '\0' na koncu string
			int k = 0;
			while ((c = getchar()) && k < 1) {

				//printf("%d \n", c);
				switch (c) {
				case 49:
					buf[0] = '1';
					break;
				case 50:
					buf[0] = '2';
					break;
				case 51:
					buf[0] = '3';
					break;
				case 52:
					buf[0] = '4';
					break;
				case 53:
					buf[0] = '5';
					break;
				default:
					printf("mozesz wybrac tylko pomiedzy 1-n \n");
					printf("ERROR : nie zapisalo zadnych danych ! wybierz dobrze \n");
					//goto END;
				}
				k++;
			}
			buf[1] = '\0';

			
			GetLocalTime(&st);
			
			if (buf[0] == '5') {
				// print some inforation only
				printf("PAMIETAJ ZE PRZY KAZDYM WYWOLANIU OKNA I WYBRANIU [5] RESETUJESZ LICZNIK \"counter\" ktory pojawia sie co %d  ms \n", TIME_SETUP);
				printf("To znaczy ze teoretycznie po sygnale rinse_alert od razu powinno byc na danej" \
					" minucie wywolane okno a wywoluja je sam kasujesz licznik co moze wplywac na zachowanie zwiazane z rinse_alert signal.\n");
				printf("counter : %d | sig_th1 : %d | rinse_alert : %d | minute : %d | (st.minute>LOOK_AHEAD_TIME) : %d\n", 
					counter, sig_th1, rinse_alert, (st.wMinute), (st.wMinute >= LOOK_AHEAD_TIME));
				printf("============================\n");
				// resetuj tylko to co bylo wywolane zeby tuitaj wejsc 
				sig_th1 = 0;
				rinse_alert = 0;
				// sleep 10 sec
				Sleep(10000);
			}

			if (buf[0] >= '1' && buf[0] <= '4') {
				// tylko dla tych scenariuszy resetuj liczniki [5] ma pokazywac tylko aktualny stan
				counter = 0;
				sig_th1 = 0;
				rinse_alert = 0;
				// zapisz tylko dla scenariuszy 1-4
				printf(" --- LOG SAVED --- \n");
				save_to_file(st, buf); // to zapisze nawet jak jest 5 < etc !!! 
			}

			printf("Oceniles swoj stan na : %s \n", buf);
			printf("rise alert status : %d %d | %d \n", rinse_alert, __is_visible(), the_signal_came);

			//counter = 0;
			//sig_th1 = 0;
			//rinse_alert = 0;

			Sleep(1000);
			ResumeThread(hThreadArray[1]);
			ShowWindow(hwnd_start, SW_MINIMIZE);

		}
	}

	return 0;
}

DWORD WINAPI func_th2(LPVOID lpParam)
{
	SYSTEMTIME st;
	while (1) {
		//printf("%d \n", counter);

		// zdarzenie ascync
		// sam pokazuje okno zeby dokonac wyboru
		// w innym przypadku okno samo sie pojawia
		if (__is_visible() == 2 && sig_th1 == 0) {
			//counter = TIME_SETUP; // up to expected value - resetowanie tutaj podowuje przy wyborze [5] zle odczyty juz nie potrzebne zerowanie
			sig_th1 = 1; // rise signal event
			rinse_alert = 1; // rise another alerts
			the_signal_came = 2; // dla BEEP zeby wiedziec skad nadeszlo wywolanie [2] - user
		}

		if (__is_visible() == 2 && (sig_th1 != 0 || rinse_alert != 0)) {
			printf(" NOT WORKING WELL %d \n", rise_error_counter);
			rise_error_signal = 1;
			rise_error_counter += 1;
		}
		
		if (counter >= TIME_SETUP) {
			sig_th1 = 1;
			the_signal_came = 1; // dla BEEP [1] app periodically call to this
			GetLocalTime(&st);
			if (st.wMinute >= LOOK_AHEAD_TIME) {
				rinse_alert = 1;
			}
		}
		if (sig_th1 == 1 && rinse_alert == 1) {
			SuspendThread(hThreadArray[1]);
			//printf("sleep 10 sec \n");
			//Sleep(10000);
		}

		// reset all coutners
		if (rise_error_counter > 10) {
			rise_error_signal = 0;
			rise_error_counter = 0;
			sig_th1 = 0;
			rinse_alert = 0;
			the_signal_came = 0;

		}

		printf(" [ counter %d ] - %d \n", counter, __is_visible());
		counter++;
		Sleep(1000);
	}
}


DWORD WINAPI func_th3(LPVOID lpParam)
{
	while (1) {
		if (sig_th1 == 1 && rinse_alert == 1) {
			Beep(1000, 500);
		}
		Sleep(1000);
	}
}



int main()
{

	HWND hwnd = GetForegroundWindow();
	hwnd_start = hwnd; // first run 

	ShowWindow(hwnd_start, SW_MINIMIZE);

	/* Initialize threads */
	//DWORD th_ids[MAX_THREADS];
	//HANDLE  hThreadArray[MAX_THREADS];

	hThreadArray[0] = CreateThread(NULL, 0, func_th1, NULL, 0, &th_ids[0]); // main
	hThreadArray[1] = CreateThread(NULL, 0, func_th2, NULL, 0, &th_ids[1]); // counters
	hThreadArray[2] = CreateThread(NULL, 0, func_th3, NULL, 0, &th_ids[2]); // beep

	WaitForMultipleObjects(MAX_THREADS, hThreadArray, TRUE, INFINITE);

	// close all threads
	for (int i = 0; i < MAX_THREADS; i++) { CloseHandle(hThreadArray[i]); }


	return 0;
}
