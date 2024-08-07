/*

  https://learn.microsoft.com/en-us/windows/win32/shutdown/how-to-shut-down-the-system

#include <windows.h>

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "advapi32.lib")

BOOL MySystemShutdown()
{
   HANDLE hToken; 
   TOKEN_PRIVILEGES tkp; 
 
   // Get a token for this process. 
 
   if (!OpenProcessToken(GetCurrentProcess(), 
        TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken)) 
      return( FALSE ); 
 
   // Get the LUID for the shutdown privilege. 
 
   LookupPrivilegeValue(NULL, SE_SHUTDOWN_NAME, 
        &tkp.Privileges[0].Luid); 
 
   tkp.PrivilegeCount = 1;  // one privilege to set    
   tkp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED; 
 
   // Get the shutdown privilege for this process. 
 
   AdjustTokenPrivileges(hToken, FALSE, &tkp, 0, 
        (PTOKEN_PRIVILEGES)NULL, 0); 
 
   if (GetLastError() != ERROR_SUCCESS) 
      return FALSE; 
 
   // Shut down the system and force all applications to close. 
 
   if (!ExitWindowsEx(EWX_SHUTDOWN | EWX_FORCE, 
               SHTDN_REASON_MAJOR_OPERATINGSYSTEM |
               SHTDN_REASON_MINOR_UPGRADE |
               SHTDN_REASON_FLAG_PLANNED)) 
      return FALSE; 

   //shutdown was successful
   return TRUE;
}
  
*/
#define _CRT_SECURE_NO_WARNINGS

#include <Windows.h>
#include <stdio.h>
#include <tchar.h>

FILE* stream_log;
volatile int counter = 0;
volatile int sig_th1 = 0;
volatile int rinse_alert = 0;
HWND hwnd_start = NULL;

#define MAX_THREADS 3
DWORD th_ids[MAX_THREADS];
HANDLE  hThreadArray[MAX_THREADS];

void save_to_file(SYSTEMTIME lt, char *retbuf)
{
	// logs 
	char log_buf[140] = { 0 };
	char log_tmp_buf[100] = { 0 };
	//strcat(log_buf, "C:\\Windows\\Temp\\v7-all"); // example file name
	strcat(log_buf, "C:\\{path_to_folder}\\{file_name}");
	strcat(log_buf, ".txt");
	printf("%s \n", log_buf);
	fopen_s(&stream_log, log_buf, "a");
	fprintf(stream_log, "%d - %02d-%02d-%02d ",  lt.wDayOfWeek, lt.wDay, lt.wMonth, lt.wYear);
	fprintf(stream_log, "%02d:%02d:%02d", lt.wHour, lt.wMinute, lt.wSecond);
	fprintf(stream_log, " | %s\n", retbuf);
	fclose(stream_log);
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

#define TIME_SETUP (60 * 5)

DWORD WINAPI func_th1(LPVOID lpParam)
{
	while (1) {
		if (counter >= (TIME_SETUP) && sig_th1 == 1) {

			HANDLE hStdout;
			hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
			cls(hStdout);

			// show window
			ShowWindow(hwnd_start, SW_RESTORE);

			printf(" some information about choice 1-3 \n");
			printf("-------------------------------------------\n");

			SYSTEMTIME st;

			int c;
			char buf[2];
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
				default:
					printf("mozesz wybrac tylko pomiedzy 1-3 \n");
					printf("ERROR : nie zapisalo zadnych danych ! wybierz dobrze \n");
					//goto END;
				}
				k++;
			}
			buf[1] = '\0';

			GetLocalTime(&st);

			save_to_file(st, buf);

			printf("Oceniles swoj stan na : %s \n", buf);
			printf("rise alert status : %d \n", rinse_alert);

			counter = 0;
			sig_th1 = 0;
			rinse_alert = 0;

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
		
		if (counter >= TIME_SETUP) {
			sig_th1 = 1;
			GetLocalTime(&st);
			if (st.wMinute >= 45) {
				rinse_alert = 1;
			}
		}
		if (sig_th1 == 1) {
			SuspendThread(hThreadArray[1]);
			//printf("sleep 10 sec \n");
			//Sleep(10000);
		}

		printf(" [ counter %d ] \n", counter);
		counter++;
		Sleep(1000);
	}
}


DWORD WINAPI func_th3(LPVOID lpParam)
{
	while (1) {
		if (sig_th1 == 1) {
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
