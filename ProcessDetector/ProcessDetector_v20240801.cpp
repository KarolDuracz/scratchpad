#define _CRT_SECURE_NO_WARNINGS

#include <windows.h>
#include <iostream>
#include <vector>
#include <psapi.h>

#define DEBUG_VAL_FOR_THREAD_COUNT 100000000
#define CNT_15_MINUTES (60 * 45) // 45 min

// boot global variables
HWND hWnd_start;
volatile int detect_event_alert = 0;

FILE* stream_log;

void save_to_file(SYSTEMTIME lt, char* retbuf, int counter, char* buf_name, DWORD pid)
{
	// logs 
	char log_buf[140] = { 0 };
	char log_tmp_buf[100] = { 0 };
	strcat(log_buf, "C:\\Users\\{path to file}\\v2-all");
	memset(log_tmp_buf, 0, 100);
	strcat(log_buf, ".txt");
	printf("%s \n", log_buf);
	fopen_s(&stream_log, log_buf, "a");
	fprintf(stream_log, "%d - %02d | %02d:%02d:%02d ", counter, lt.wDayOfWeek, lt.wDay, lt.wMonth, lt.wYear);
	fprintf(stream_log, "%02d:%02d:%02d", lt.wHour, lt.wMinute, lt.wSecond);
	fprintf(stream_log, " | [%d] %s | ", pid, buf_name);
	fprintf(stream_log, " %s\n", retbuf);
	fclose(stream_log);
}

void encode_wchar_to_str(char* buffer, char* retbuf, int len)
{
	int j;
	int jj = 0;
	for (j = 0; j < len * 2; j++) {
		if (buffer[j] == 0) continue;
		//printf("%c", buffer[j]);
		retbuf[jj] = buffer[j];
		jj++;
	}
	retbuf[jj++] = '\0';
}

void show_message(HWND hWnd)
{
	//MessageBox(hWnd, L"Czy właśnie otworzyłeś okno Extensions / Rozszerzenia w przeglądarce Edge lub Chrome?!", L"Alert!", MB_OK);
	// https://stackoverflow.com/questions/31563579/execute-command-using-win32
	system("notepad.exe C:\\Users\\{path to file}\\v2-all.txt");
}

/* main thread */
DWORD WINAPI func_th1(LPVOID lpParam)
{

#define BUF_SIZ 500
	char buf_name[BUF_SIZ];
	char* buffer = new char[BUF_SIZ];
	char* retbuf = new char[BUF_SIZ];
	DWORD pid;

	std::vector<std::string> patter_list;
	patter_list.push_back("xtensions");
	patter_list.push_back("Rozszerzenia");

	SYSTEMTIME st;

	//int counter = 0;
	while (1) {
		/*if (counter % DEBUG_VAL_FOR_THREAD_COUNT == 0) {
			std::cout << " [1] " << counter << std::endl;
		}*/

		HWND hWnd = GetForegroundWindow();

		GetWindowTextW(hWnd, (LPWSTR)buffer, BUF_SIZ);
		int len = GetWindowTextLengthA(hWnd);
		GetWindowThreadProcessId(hWnd, &pid);
		encode_wchar_to_str(buffer, retbuf, len);

		std::string str_retbuf(retbuf, len);
		//std::cout << str_retbuf.find(0x7) << "|" << (str_retbuf.find(0x7) < (1<<31)) << "|" << retbuf << std::endl;
		if ((str_retbuf.find(0x7) < (1 << 31)) == 1) {
			std::cout << " some application name but with beep sound in cmd.exe " << std::endl;
		}
		else {
			std::cout <<  retbuf << std::endl;
		}

		GetLocalTime(&st);

		HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_TERMINATE, FALSE, pid);
		DWORD dump_pid_name = GetProcessImageFileNameA(hProcess, (LPSTR)buf_name, BUF_SIZ);

		std::string str(retbuf, len);

		int jj = 0;
		for (int i = 0; i < patter_list.size(); i++) {
			if (str.find(patter_list[i]) >= 0 && str.find(patter_list[i]) <= jj || str.find(patter_list[i]) <= sizeof(retbuf)) {
				std::cout << " >>>>>>>>>>>>>>>>>>>>>>>>>> " << str << " " << str.find(patter_list[i]) << std::endl;
				detect_event_alert = 1;
				save_to_file(st, retbuf, i, buf_name, pid);
				show_message(hWnd);
			}
		}

		Sleep(1000);

		//counter++;
	}
	return 0;
}

/* thread for counters */
DWORD WINAPI func_th2(LPVOID lpParam)
{
	int counter = 0;
	while (1) {
		/*if (counter % DEBUG_VAL_FOR_THREAD_COUNT == 0) {
			std::cout << " [2] " << counter << std::endl;
		}*/
		if (counter >= CNT_15_MINUTES) {
			std::cout << " [2] " << counter << std::endl;
			system("calc.exe");
			counter = 0;
		}
		std::cout << " [conter test] " << counter << "|" << CNT_15_MINUTES << std::endl;
		counter++;
		Sleep(1000);
	}
	return 0;
}


/* thread IsIconic */
DWORD WINAPI func_th3(LPVOID lpParam)
{
	int counter = 0;
	int is_iconic_counter = 0;
	DWORD pid;
	while (1) {
		GetWindowThreadProcessId(hWnd_start, &pid);
		std::cout << " hWnd_start >> " << hWnd_start << "|" << pid << "|||" << IsIconic(hWnd_start) << std::endl;
		if (detect_event_alert == 1) {
			if (IsIconic(hWnd_start) == 1) {
				ShowWindow(hWnd_start, SW_SHOW | SW_RESTORE);
			}
			MessageBox(hWnd_start, L"Czy właśnie otworzyłeś okno Extensions / Rozszerzenia w przeglądarce Edge lub Chrome?!", L"Alert!", MB_OK);
			detect_event_alert = 0;
		}
		if (IsIconic(hWnd_start) == 0) {
			if (is_iconic_counter >= 3) {
				ShowWindow(hWnd_start, SW_MINIMIZE);
				is_iconic_counter = 0;
			}
			is_iconic_counter++;
		}
		Sleep(1000);
	}
	return 0;
}

#define MAX_THREADS 3
#define DEBUG // if this no exist then dont run code below

int main()
{
	/* boot app start sequence */
	HWND hWnd = GetForegroundWindow();
	hWnd_start = hWnd; // first run 
	//SetWindowPos(hWnd, NULL, 400, 400, 100, 100, SWP_HIDEWINDOW);
	SetWindowPos(hWnd, NULL, 0, 0, 700, 900, SWP_SHOWWINDOW);
#ifndef DEBUG
		ShowWindow(hWnd, SW_MINIMIZE);
		/* disable user inputs for hwnd*/
		EnableWindow(hWnd_start, FALSE); // user can't close window cliced on "X" on the right top corner etc. Only PE etc can close process
#endif
	/* write to log system app time running */
	SYSTEMTIME st;
	GetLocalTime(&st);
	save_to_file(st, (char*)"APP IS RUNNING", 0, (char*)"=== start application system for detect process ===", 0);

	/* Initialize threads */
	DWORD th_ids[MAX_THREADS];
	HANDLE  hThreadArray[MAX_THREADS];

	hThreadArray[0] = CreateThread(NULL, 0, func_th1, NULL, 0, &th_ids[0]); // main
	hThreadArray[1] = CreateThread(NULL, 0, func_th2, NULL, 0, &th_ids[1]); // counters
	hThreadArray[2] = CreateThread(NULL, 0, func_th3, NULL, 0, &th_ids[2]); // isiconic

	WaitForMultipleObjects(MAX_THREADS, hThreadArray, TRUE, INFINITE);

	// close all threads
	for (int i = 0; i < MAX_THREADS; i++) { CloseHandle(hThreadArray[i]); }

	return 0;
}
