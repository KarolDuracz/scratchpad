#include <Windows.h>
#include <stdio.h>
#include <strsafe.h>
#include <string>

#define BUFFERSIZE 200

VOID CALLBACK FileIOCompletionRoutine(
	__in  DWORD dwErrorCode,
	__in  DWORD dwNumberOfBytesTransfered,
	__in  LPOVERLAPPED lpOverlapped
)
{

}


int main()
{

	char   ReadBuffer[BUFFERSIZE] = { 0 };
	OVERLAPPED ol = { 0 };

	system("tasklist | findstr /i \"ProcessTrackerHelp\" > c:\\Windows\\Temp\\dddump.txt");

	HANDLE hFile;
	hFile = CreateFile(L"c:\\Windows\\Temp\\dddump.txt", GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
		FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, NULL);

	if (hFile == INVALID_HANDLE_VALUE) {
		// save to log
	}

	if (FALSE == ReadFileEx(hFile, ReadBuffer, BUFFERSIZE - 1, &ol, FileIOCompletionRoutine))
	{
		CloseHandle(hFile);
		return -1;
	}

	// reading
	printf("%s \n", ReadBuffer);
	int count_chars = 0;
	for (int i = 0; i < BUFFERSIZE; i++) {
		if (ReadBuffer[i] != 0) {
			count_chars++;
		}
		printf("%d ", ReadBuffer[i]);
	}
	printf("counted chars %d \n", count_chars);
	//ReadBuffer[dwBytesRead] = '\0';
	//printf("%s\n", ReadBuffer);

	int count_space = 0;
	char buf[50] = { 0 };
	int k = 0;
	if (count_chars != 0) {
		for (int i = 0; i < count_chars; i++) {
			// find first space
			if (ReadBuffer[i] == 32) {
				count_space++;
			}
			if (count_space >= 5 && count_space < 6) {	// this particular pattern for cut number from string
				buf[k] = ReadBuffer[i];
				printf("%d \n", buf[k]);
				k++;			
			}
		}
	}

	buf[k + 1] = '\0';
	printf(" end %s \n", buf);

	CloseHandle(hFile);

	// write to file PID 

	HANDLE hFile2;
	hFile2 = CreateFile(L"c:\\Windows\\Temp\\dddump_PID.txt", GENERIC_WRITE, 0, NULL, CREATE_ALWAYS,
		FILE_ATTRIBUTE_NORMAL, NULL);

	if (hFile2 == INVALID_HANDLE_VALUE) {
		// save to log
	}

	DWORD writtenbytes;
	BOOL wrFlag = WriteFile(hFile2, buf, sizeof(buf), &writtenbytes, NULL);
	printf("written bytes %d \n", writtenbytes);

	return 0;
}
