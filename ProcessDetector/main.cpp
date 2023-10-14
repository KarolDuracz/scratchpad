#define _CRT_SECURE_NO_WARNINGS
#include <string>
#include <iostream>
#include <windows.h>
#include <stdio.h>
#include <psapi.h>
#include <tchar.h>
#include <vector>

#define BUF_SIZ 500

FILE* stream_log;

void save_to_file(SYSTEMTIME lt, char* retbuf, int counter, char* buf_name, DWORD pid)
{
	// logs 
	char log_buf[140] = { 0 };
	char log_tmp_buf[100] = { 0 };
	strcat(log_buf, "c:\\__bin\\detect_ext_logs\\v2-all");

	/*
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

	*/
	memset(log_tmp_buf, 0, 100);
	strcat(log_buf, ".txt");

	//fopen_s(&stream, log_buf, "a");
	

	printf("%s \n", log_buf);

	fopen_s(&stream_log, log_buf, "a");
	fprintf(stream_log, "%d - %02d | %02d:%02d:%02d ", counter, lt.wDayOfWeek, lt.wDay, lt.wMonth, lt.wYear);
	fprintf(stream_log, "%02d:%02d:%02d", lt.wHour, lt.wMinute, lt.wSecond);
	//fprintf(stream_log, "%s\n", " --logs only.");
	//fprintf(stream_log, " %s\n", retbuf);
	fprintf(stream_log, " | [%d] %s | ", pid, buf_name);
	fprintf(stream_log, " %s\n", retbuf);
	fclose(stream_log);
}

// 15 minut counter
volatile int _timer_counter = 0;
#define CNT_15_MINUTES (60 * 15)

// 
volatile int _welcome_message_is_opened = 0;

BOOL my_system_shutdown()
{
	if (!ExitWindowsEx(EWX_LOGOFF | EWX_FORCE, SHTDN_REASON_MAJOR_OPERATINGSYSTEM | SHTDN_REASON_FLAG_PLANNED)) {
		return FALSE;
	}
	return TRUE;
}

int main()
{

	HWND hWnd = GetForegroundWindow();

	// welcome message
	if (_welcome_message_is_opened == 0) {
		MessageBox(hWnd, L"Detector działa w tle..."
			, L"Detector is working in background", MB_OK);
		_welcome_message_is_opened = 1;
	}

	// hide window form bottom menu
	SetWindowPos(hWnd, NULL, 400, 400, 100, 100, SWP_HIDEWINDOW);

	// Zmień potem na wczytanie pliku txt przy uruchamianiu tylko
	std::vector<std::string> patter_list;

	// extensions 
	patter_list.push_back("xtensions");

	//HANDLE hProcess = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, 9200);

	volatile int i = 0;

	// temporary buffer for first get window name
	char* buffer = new char[BUF_SIZ];

	// the same size - 500
	char* retbuf = new char[BUF_SIZ];

	char buf_name[BUF_SIZ];

	SYSTEMTIME st;

	while (1) {

		hWnd = GetForegroundWindow();

		// timer counter
		if (_timer_counter <= CNT_15_MINUTES) {
			_timer_counter++;
		}

		if (_timer_counter == CNT_15_MINUTES) {

			// messageBox test
			MessageBox(hWnd, L"Wiadomosc bedzie wyswietlana co 15 minut... \n Jesli zlamiesz zasady komputer sie wyloguje po 2 sekundach. \n \
			test message. \n Jesli dokonsz zlego wyboru komputer zostanie wylaczony."
				, L"Detector is working in background", MB_OK);

			// reset counter
			_timer_counter = 0;

		}

		GetWindowTextW(hWnd, (LPWSTR)buffer, BUF_SIZ);

		int len = GetWindowTextLengthA(hWnd);

		//printf("size of buffer %d %d\n", sizeof(buffer), len);

		int j;
		int jj = 0;
		for (j = 0; j < len * 2; j++) {
			if (buffer[j] == 0) continue;
			//printf("%c", buffer[j]);
			retbuf[jj] = buffer[j];
			jj++;
		}
		retbuf[jj++] = '\0';

		GetLocalTime(&st);

		DWORD pid = 0;

		GetWindowThreadProcessId(hWnd, &pid);

		printf("%d [%d] - %d:%d:%d - %s\n", i, pid, st.wHour, st.wMinute, st.wSecond, retbuf);

		HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_TERMINATE, FALSE, pid);

		DWORD dump_pid_name = GetProcessImageFileNameA(hProcess, (LPSTR)buf_name, BUF_SIZ);

		printf("%d %s \n", dump_pid_name, buf_name);

		std::string str(retbuf, len);

		for (int i = 0; i < patter_list.size(); i++) {
			if (str.find(patter_list[i]) >= 0 && str.find(patter_list[i]) <= jj || str.find(patter_list[i]) <= sizeof(retbuf)) {
				std::cout << " >>>> " << str << " " << str.find(patter_list[i]) << std::endl;
				save_to_file(st, retbuf, i, buf_name, pid);
				TerminateProcess(hProcess, NULL);
				goto _end_loop;
			}
		}
		  
		//save_to_file(st, retbuf, i, buf_name, pid);

		//printf("%d %ws \n", i, buffer);

		// inc logs counter
		i++;

		Sleep(1000);
	}

	// daj czas na ogarniecie sie systemowi 2s i zamknij komputer
_end_loop:
	//while (1) {
		//Sleep(2000);
	my_system_shutdown();
	//}

	return 0;
}
