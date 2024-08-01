#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <fstream>
#include <iostream>
#include <string>
#include <ctime>

#include <stdio.h>
#include <strsafe.h>

#define BUFFERSIZE 200

VOID CALLBACK FileIOCompletionRoutine(
    __in  DWORD dwErrorCode,
    __in  DWORD dwNumberOfBytesTransfered,
    __in  LPOVERLAPPED lpOverlapped
)
{

}


SERVICE_STATUS        g_ServiceStatus = { 0 };
SERVICE_STATUS_HANDLE g_StatusHandle = NULL;
HANDLE                g_ServiceStopEvent = INVALID_HANDLE_VALUE;

void WINAPI ServiceMain(DWORD argc, LPTSTR* argv);
void WINAPI ServiceCtrlHandler(DWORD);

#define SERVICE_NAME  ((LPWSTR)"SimpleService")

volatile int global_counter = 0;

// Function to log messages to a file
void LogMessage(const std::string& message) {
    std::ofstream logFile;
    logFile.open("C:\\Windows\\Temp\\service.log", std::ios_base::app); // Adjust the path as needed
    if (logFile.is_open()) {
        logFile << message << std::endl;
        logFile.close();
    }
}

int main() {
    SERVICE_TABLE_ENTRY ServiceTable[] =
    {
        {SERVICE_NAME, (LPSERVICE_MAIN_FUNCTION)ServiceMain},
        {NULL, NULL}
    };

    if (StartServiceCtrlDispatcher(ServiceTable) == FALSE) {
        return GetLastError();
    }

    return 0;
}

void WINAPI ServiceMain(DWORD argc, LPTSTR* argv) {
    DWORD Status = E_FAIL;

    g_StatusHandle = RegisterServiceCtrlHandler(SERVICE_NAME, ServiceCtrlHandler);

    if (g_StatusHandle == NULL) {
        goto EXIT;
    }

    // Initialize the service status structure
    g_ServiceStatus.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
    g_ServiceStatus.dwControlsAccepted = 0;
    g_ServiceStatus.dwCurrentState = SERVICE_START_PENDING;
    g_ServiceStatus.dwWin32ExitCode = 0;
    g_ServiceStatus.dwServiceSpecificExitCode = 0;
    g_ServiceStatus.dwCheckPoint = 0;

    if (SetServiceStatus(g_StatusHandle, &g_ServiceStatus) == FALSE) {
        goto EXIT;
    }

    // Create a service stop event to wait on later
    g_ServiceStopEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
    if (g_ServiceStopEvent == NULL) {
        goto EXIT;
    }

    // Report running status when initialization is complete
    g_ServiceStatus.dwControlsAccepted = SERVICE_ACCEPT_STOP;
    g_ServiceStatus.dwCurrentState = SERVICE_RUNNING;
    g_ServiceStatus.dwWin32ExitCode = 0;
    g_ServiceStatus.dwCheckPoint = 0;

    if (SetServiceStatus(g_StatusHandle, &g_ServiceStatus) == FALSE) {
        goto EXIT;
    }

    // Log service start
    LogMessage("Service started.");

    // Main service loop
    while (WaitForSingleObject(g_ServiceStopEvent, 0) != WAIT_OBJECT_0) {
        // Perform main service function here
        // Example: Write current timestamp to log file

        std::time_t now = std::time(nullptr);
        std::string timestamp = std::ctime(&now);
        timestamp.pop_back(); // Remove trailing newline
        LogMessage("Current timestamp: " + timestamp + " | " + std::to_string(global_counter));

        Sleep(1000); // Wait for 1 second

        if (global_counter == 0) {
        
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

            CloseHandle(hFile2);

            // last step
            if (count_chars != 0) {
                global_counter++; // inc global_counter
            }
        }
    }

    // Log service stop
    LogMessage("Service stopped.");

    // Stop the service
    g_ServiceStatus.dwControlsAccepted = 0;
    g_ServiceStatus.dwCurrentState = SERVICE_STOP_PENDING;
    g_ServiceStatus.dwWin32ExitCode = 0;
    g_ServiceStatus.dwCheckPoint = 3;

    if (SetServiceStatus(g_StatusHandle, &g_ServiceStatus) == FALSE) {
        goto EXIT;
    }

EXIT:
    if (g_ServiceStopEvent) {
        CloseHandle(g_ServiceStopEvent);
    }

    g_ServiceStatus.dwControlsAccepted = 0;
    g_ServiceStatus.dwCurrentState = SERVICE_STOPPED;
    g_ServiceStatus.dwWin32ExitCode = Status;
    g_ServiceStatus.dwCheckPoint = 0;

    if (g_StatusHandle) {
        SetServiceStatus(g_StatusHandle, &g_ServiceStatus);
    }
    return;
}

void WINAPI ServiceCtrlHandler(DWORD CtrlCode) {
    switch (CtrlCode) {
    case SERVICE_CONTROL_STOP:
        if (g_ServiceStatus.dwCurrentState != SERVICE_RUNNING)
            break;

        g_ServiceStatus.dwControlsAccepted = 0;
        g_ServiceStatus.dwCurrentState = SERVICE_STOP_PENDING;
        g_ServiceStatus.dwWin32ExitCode = 0;
        g_ServiceStatus.dwCheckPoint = 4;

        SetServiceStatus(g_StatusHandle, &g_ServiceStatus);

        // Signal the service to stop
        SetEvent(g_ServiceStopEvent);
        break;

    default:
        break;
    }
}
