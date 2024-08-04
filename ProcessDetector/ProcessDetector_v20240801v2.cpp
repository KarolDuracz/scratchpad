#define _CRT_SECURE_NO_WARNINGS

#include <windows.h>
#include <iostream>
#include <vector>
#include <psapi.h>

#include <conio.h>
#include <tchar.h>

#include <string>
#include <codecvt>
#include <locale>
#include <unordered_map>


#define MAX_THREADS 3
#define DEBUG // if this no exist then dont run code below

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
	strcat(log_buf, "C:\\Users\\kdhome\\Documents\\progs\\__bin\\__process_detector_in_detox_2024\\v2-all");
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

#if 0
// Function to perform transliteration from Cyrillic to Latin
std::string transliterate(const std::wstring& text) {
	// Mapping of Cyrillic characters to Latin characters
	std::unordered_map<wchar_t, std::string> cyrillic_to_latin = {
		{L'А', "A"}, {L'Б', "B"}, {L'В', "V"}, {L'Г', "G"}, {L'Д', "D"}, {L'Е', "E"}, {L'Ё', "YO"},
		{L'Ж', "ZH"}, {L'З', "Z"}, {L'И', "I"}, {L'Й', "Y"}, {L'К', "K"}, {L'Л', "L"}, {L'М', "M"},
		{L'Н', "N"}, {L'О', "O"}, {L'П', "P"}, {L'Р', "R"}, {L'С', "S"}, {L'Т', "T"}, {L'У', "U"},
		{L'Ф', "F"}, {L'Х', "KH"}, {L'Ц', "TS"}, {L'Ч', "CH"}, {L'Ш', "SH"}, {L'Щ', "SCH"}, {L'Ъ', ""},
		{L'Ы', "Y"}, {L'Ь', ""}, {L'Э', "E"}, {L'Ю', "YU"}, {L'Я', "YA"},
		{L'а', "a"}, {L'б', "b"}, {L'в', "v"}, {L'г', "g"}, {L'д', "d"}, {L'е', "e"}, {L'ё', "yo"},
		{L'ж', "zh"}, {L'з', "z"}, {L'и', "i"}, {L'й', "y"}, {L'к', "k"}, {L'л', "l"}, {L'м', "m"},
		{L'н', "n"}, {L'о', "o"}, {L'п', "p"}, {L'р', "r"}, {L'с', "s"}, {L'т', "t"}, {L'у', "u"},
		{L'ф', "f"}, {L'х', "kh"}, {L'ц', "ts"}, {L'ч', "ch"}, {L'ш', "sh"}, {L'щ', "sch"}, {L'ъ', ""},
		{L'ы', "y"}, {L'ь', ""}, {L'э', "e"}, {L'ю', "yu"}, {L'я', "ya"}
	};

	// Mapping of Cyrillic characters to Polish Latin characters
	/*
	std::unordered_map<wchar_t, std::string> cyrillic_to_polish = {
		{L'А', "A"}, {L'Б', "B"}, {L'В', "W"}, {L'Г', "G"}, {L'Д', "D"}, {L'Е', "E"}, {L'Ё', "Jo"},
		{L'Ж', "Ż"}, {L'З', "Z"}, {L'И', "I"}, {L'Й', "J"}, {L'К', "K"}, {L'Л', "L"}, {L'М', "M"},
		{L'Н', "N"}, {L'О', "O"}, {L'П', "P"}, {L'Р', "R"}, {L'С', "S"}, {L'Т', "T"}, {L'У', "U"},
		{L'Ф', "F"}, {L'Х', "Ch"}, {L'Ц', "C"}, {L'Ч', "Cz"}, {L'Ш', "Sz"}, {L'Щ', "Szcz"}, {L'Ъ', ""},
		{L'Ы', "Y"}, {L'Ь', ""}, {L'Э', "E"}, {L'Ю', "Ju"}, {L'Я', "Ja"},
		{L'а', "a"}, {L'б', "b"}, {L'в', "w"}, {L'г', "g"}, {L'д', "d"}, {L'е', "e"}, {L'ё', "jo"},
		{L'ж', "ż"}, {L'з', "z"}, {L'и', "i"}, {L'й', "j"}, {L'к', "k"}, {L'л', "l"}, {L'м', "m"},
		{L'н', "n"}, {L'о', "o"}, {L'п', "p"}, {L'р', "r"}, {L'с', "s"}, {L'т', "t"}, {L'у', "u"},
		{L'ф', "f"}, {L'х', "ch"}, {L'ц', "c"}, {L'ч', "cz"}, {L'ш', "sz"}, {L'щ', "szcz"}, {L'ъ', ""},
		{L'ы', "y"}, {L'ь', ""}, {L'э', "e"}, {L'ю', "ju"}, {L'я', "ja"}
	};
	*/

	std::string result;
	for (const wchar_t& ch : text) {
		if (cyrillic_to_latin.find(ch) != cyrillic_to_latin.end()) {
			result += cyrillic_to_latin[ch];
		}
		else {
			result += ch; // if character is not in the map, add it as is
		}
	}
	return result;
}

template<class Facet>
struct deletable_facet : Facet
{
	template<class... Args>
	deletable_facet(Args&&... args) : Facet(std::forward<Args>(args)...) {}
	~deletable_facet() {}
};

void track_and_collect_names(std::string str)
{
	//std::cout << "[call from track_and_collect_names |" <<  __FUNCTION__ << str << std::endl;
	std::wstring wc = std::wstring(str.begin(), str.end());
	const wchar_t* out = wc.c_str();
	
	//printf("[ >>> ] %ws \n", out);

	std::locale russian_locale("ru_RU.UTF-8");

	//std::wcin.imbue(std::locale());
	std::wcout.imbue(russian_locale);

	std::wstring russian_text = L"Привет, мир!";
	//std::wstring russian_text = out;
	std::wcout << L"Original text: " << russian_text << std::endl;

	std::string transliterated_text = transliterate(russian_text);
	std::cout << "Transliterated text: " << transliterated_text << std::endl;

	//std::wstring russian_text = L"Привет, мир!";

	// Print using std::wcout with the Russian locale
	//std::wcout.imbue(russian_locale);
	//std::wcout << russian_text << std::endl;

	// Converting wide string to narrow string using std::wstring_convert
	//std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
	//std::string narrow_text = converter.to_bytes(russian_text);

	// Print the narrow string using std::cout
	//std::cout << narrow_text << std::endl;

	//std::wcout << ">>> " << str.c_str() << std::locale("").name().c_str() << "\n" << std::endl;

	/*std::wstring_convert<deletable_facet<std::codecvt<char16_t, char, std::mbstate_t>>, char16_t > converter;
	std::string data = reinterpret_cast<const char*>(str.c_str());
	std::u16string aDst = converter.from_bytes(data);

	std::cout << aDst.c_str() << std::endl;*/
	
}
#endif

/* --- */
int temp_buf_for_unique_hash[10000];
int current_hash = 0;
int prev_hash = 0;
volatile int idx_unique_hash = 0;
volatile int hh = 0;
std::vector<int> vec_hash;
/* --- */

struct STRING_DATA_DB {
	const char *name; // strdup --> using assert(strcmp(s1, s2) == 0) before alloc memory
	int hash; // at this moment NULL
	int tick_measure; // in seconds
	// etc in progress...
};

#define HASHSIZE 10000 // at this moment for debug 100 but in future 1000 or more
static struct STRING_DATA_DB hashtab[HASHSIZE];

unsigned hash(char* s)
{
	unsigned hashval;
	for (hashval = 0; *s != '\0'; s++) {
		hashval = *s + 31 * hashval;
	}
	return hashval % HASHSIZE;
}

typedef struct {
	int start;
	int end;
} INDEX_OF_CUTOUT;

int check_is_digit(int val)
{
	if (val >= 48 && val <= 57) {
		return 1;
	}
	return -1;
}

#define SCENARIO_STRONY 0
#define SCENARIO_STRON  1
#define SCENARIO_ONETAB  2
void concat_name(std::string str)
{
	std::cout << " find > " << str.find("i jeszcze") << " " << str.find("strony") << std::endl;

	int scenario = SCENARIO_STRONY;

	// find which scenario
	if (str.find("i jeszcze") < (1 << 31)) {
		if (str.find("strony")) {
			scenario = SCENARIO_STRONY;
		}
		if (str.find("stron")) {
			scenario = SCENARIO_STRON;
		}
	}

	std::cout << " selected scanario is : " << scenario << std::endl;

	char buf[3];
	int index = 0;
	int k = 0;

	ZeroMemory(buf, 3); // init memory for buf 

	INDEX_OF_CUTOUT icc = { 0 };

	switch (scenario) {
	case SCENARIO_STRONY:
		if (str.find("i jeszcze") < (1 << 31)) {
			for (int i = str.find("i jeszcze"); i < (strlen("strony") + str.find("strony")); i++) {
				//std::cout << str[i] << " " << check_is_digit(str[i]) << std::endl;
				if (check_is_digit(str[i]) == 1) {
					buf[index] = str[i];
					index++;
				}
				// inc k counter for INDEX_OF_CUTOUT
				k++;
			}
			icc.start = str.find("i jeszcze");
			icc.end = str.find("i jeszcze") + k;
		}
		break;

	case SCENARIO_STRON:
		if (str.find("i jeszcze") < (1 << 31)) {
			for (int i = str.find("i jeszcze"); i < (strlen("stron") + str.find("stron")); i++) {
				//std::cout << str[i] << " " << check_is_digit(str[i]) << std::endl;
				if (check_is_digit(str[i]) == 1) {
					buf[index] = str[i];
					index++;
				}
				// inc k counter for INDEX_OF_CUTOUT
				k++;
			}
			icc.start = str.find("i jeszcze");
			icc.end = str.find("i jeszcze") + k;
		}
		break;

	case SCENARIO_ONETAB:
		// scenariusz dla tylko 1 karty np Youtube wtedy nie wykona tego co wyżej
		// trzeba to obłsużyć też
		break;

	}

	buf[index++] = '\0';

	if (icc.start != 0) {
		std::cout << " BUF " << buf[0] << buf[1] << buf[2] << "|" << str[icc.start] << ":" << str[icc.end-1] << "|" << icc.start << std::endl;
	}

	char new_str[500];
	int i;
	int j = 0;
	// concatenate new string 
	for (i = 0; i < strlen(str.c_str()); i++) {
		if (i >= icc.start && i <= icc.end - 1)
			continue;
		new_str[j] = str[i];
		j++;
	}
	new_str[j] = '\0';

	// i ten string dopiero można zamienić jako HASH żeby wrzucać jako struktura do HASHTAB
	// tylko w przypadku codziennego użytkownia musi być zapisane w pliku .txt np cała lista ID 
	// żeby nie generować co chwile innych hashów, ale w sumie nie powinno tak być jeśli wezmę z ANSIC implementację.
	std::cout << " NEW STRING " << new_str << " | HASH | " << hash(new_str) << std::endl;

	// ok this is for test 
	// zanim zrobie jakies konkrety z hashtab sprwadze ile jest ogólnie unikalnych id (hashów)
	// na początek 500 wielkosc bufora
	/*for (int i = 0; i < 500; i++) {
		if (temp_buf_for_unique_hash[i] != hash(new_str)) {
			temp_buf_for_unique_hash[i] = hash(new_str);
			break;
		}
	}*/

	//if (idx_unique_hash == 0) {
	//	temp_buf_for_unique_hash[idx_unique_hash] = 1;
	//	idx_unique_hash++;
	//}

	//current_hash = hash(new_str);
	//if (current_hash != prev_hash) {
	//	// add to arr
	//	for (int i = 0; i < idx_unique_hash; i++) {
	//		if (temp_buf_for_unique_hash[i] != hash(new_str)) {
	//			std::cout << " ================ " << std::endl;
	//			goto _END;
	//		}
	//	}
	//_END:
	//	std::cout << " JUM TO END " << std::endl;
	//	temp_buf_for_unique_hash[idx_unique_hash] = hash(new_str);
	//	idx_unique_hash++;
	//}
	//prev_hash = current_hash;

	//int kk = 0;
	//for (int i = 0; i < 500; i++) {
	//	if (temp_buf_for_unique_hash[i] != 0) {
	//		kk++;
	//	}
	//}

	//std::cout << " UNIQUE ID's : " << kk << std::endl;

	// efektywne to nie jest bo 10.000 elementow przeszukiwac co chwila gdy jest raptem 10-20 albo 100 max
	// to nie tylko zzera pamiec ale i czas procesora... no ale na ten moment jedyne co mi dziala ;/
	// jeszcze do tego 2x to samo 10.000 elementow najpierw iterując "i", a potem tablice hashów heh

	current_hash = hash(new_str);
	std::cout << " CHECK IF CMP CURR == PREV : " << (current_hash != prev_hash) << std::endl;
	if ((current_hash != prev_hash)) {
		for (int i = 0; i < HASHSIZE; i++) {
			if (i == hash(new_str)) {
				temp_buf_for_unique_hash[i] = 1;
			}
		}

		hh = 0;
		vec_hash.clear(); // clear array
		for (int i = 0; i < HASHSIZE; i++) {
			if (temp_buf_for_unique_hash[i] == 1) {
				vec_hash.push_back(i);
				hh++;
			}
		}
	}
	prev_hash = current_hash;

	std::cout << " UNIQUE ID's : " << hh << std::endl;
	std::cout << " vec size : " << vec_hash.size() << std::endl;

	for (int val : vec_hash) {
		std::cout << val << ' ';
	}
	std::cout << '\n';

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
	system("notepad.exe C:\\Users\\kdhome\\Documents\\progs\\__bin\\__process_detector_in_detox_2024\\v2-all.txt");
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

		/* collect some data */
		// in sequence this might be higher but for this application this is not matter where plase for str, and call to this function and
		// what is after that. 
		// in development...
		//track_and_collect_names(str);
		concat_name(str);

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
#ifndef DEBUG
				ShowWindow(hWnd_start, SW_MINIMIZE);
#endif
				is_iconic_counter = 0;
			}
			is_iconic_counter++;
		}
		Sleep(1000);
	}
	return 0;
}


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
