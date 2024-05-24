#include <windows.h>
#include <stdio.h>

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

volatile int counter = 0;;

void __get_local_time()
{
	SYSTEMTIME st;
	GetLocalTime(&st);
	printf("%02d | %02d:%02d:%02d - %d:%d:%d\n", st.wDayOfWeek, st.wDay, st.wMonth, st.wYear, st.wHour, st.wMinute, st.wSecond);
}

int main()
{

	while (1) {
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

		static char buf[100];
		static char _cur_buf[100];
		GetWindowTextA(hwnd, buf, 100);
		//printf("[1] %s %d %d %d %s \n", buf, x1, x2, x3, _cur_buf);

		int cmp_ret = strcmp(buf, _cur_buf);
		//printf("string cmp %d \n", cmp_ret);

		// jesli nie jest rowne zero -1 / 1 wtedy zrob funkcje na lockach i bardziej wymagajace 
		if (cmp_ret != 0) {
			printf("%d %s %s\n", counter, buf, _cur_buf);
			__get_local_time();
			counter = 0;
		}
		else {
			printf(".");
			counter++;
		}

		strcpy_s(_cur_buf, 100, buf);

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
