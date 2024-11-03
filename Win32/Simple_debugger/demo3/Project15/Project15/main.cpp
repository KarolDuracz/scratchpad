#include <windows.h>
#include <iostream>
#include <tlhelp32.h>
#include <stdio.h>
#include <tchar.h>
#include <stdlib.h>

unsigned char customThreadFunc[] = {
    // This is machine code for a simple function.
    // In this example, it will set a value in memory.

    0xC7, 0x45, 0x00, 0x01, 0x00, 0x00, 0x00,  // mov DWORD PTR [ebp], 1
    0xEB, 0xFE                                 // jmp $
    // This is an infinite loop.
};

// Function to get PID of a running process by its name
DWORD GetProcessIdByName(const char* processName) {
    PROCESSENTRY32 entry;
    entry.dwSize = sizeof(PROCESSENTRY32);
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);

    //unsigned char* p = (unsigned char*)malloc(100); 
    wchar_t wszBuff[100];
    MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, processName, -1, wszBuff, 100);
    printf("%ws %d\n", wszBuff, wcslen(wszBuff));

    if (Process32First(snapshot, &entry)) {
        while (Process32Next(snapshot, &entry)) {
            //std::cout << entry.th32ProcessID << std::endl;
            _tprintf(entry.szExeFile);
            printf(" | PID - string len : [ %d ] - %d \n", entry.th32ProcessID, wcslen(entry.szExeFile));
            //mbstowcs(p, entry.szExeFile, wcslen(entry.szExeFile));
            std::cout << " compare : " << wcscmp(wszBuff, entry.szExeFile) << std::endl;
            if (wcscmp(wszBuff, entry.szExeFile) == 0) {
                CloseHandle(snapshot);
                return entry.th32ProcessID;
            }
        }
    }
    CloseHandle(snapshot);
    return 0;
}

bool EnableDebugPrivilege(HANDLE process) {
    HANDLE hToken;
    LUID luid;
    TOKEN_PRIVILEGES tokenPrivileges;

    // Open the current process token
    if (!OpenProcessToken(/*GetCurrentProcess()*/ process, TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken)) {
        std::cerr << "Failed to open process token: " << GetLastError() << std::endl;
        return false;
    }

    // Get the LUID for SeDebugPrivilege
    if (!LookupPrivilegeValue(NULL, SE_DEBUG_NAME, &luid)) {
        std::cerr << "LookupPrivilegeValue error: " << GetLastError() << std::endl;
        CloseHandle(hToken);
        return false;
    }

    // Set up privilege adjustment
    tokenPrivileges.PrivilegeCount = 1;
    tokenPrivileges.Privileges[0].Luid = luid;
    tokenPrivileges.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

    // Adjust token privileges
    if (!AdjustTokenPrivileges(hToken, FALSE, &tokenPrivileges, sizeof(TOKEN_PRIVILEGES), NULL, NULL)) {
        std::cerr << "AdjustTokenPrivileges error: " << GetLastError() << std::endl;
        CloseHandle(hToken);
        return false;
    }

    CloseHandle(hToken);
    return true;
}

// Define a struct to hold the parameters
struct ThreadParams {
    HWND hWnd;        // Window handle
    LPCSTR lpText;    // Message text
    LPCSTR lpCaption; // Message caption
    UINT uType;       // Message box type
};

// Custom thread routine that will be executed in the new thread
DWORD WINAPI CustomThreadRoutine(LPVOID lpParam) {
    // Cast the parameter to the correct type
    ThreadParams* params = (ThreadParams*)lpParam;

    // Call MessageBoxA using the parameters from the struct
    MessageBoxA(params->hWnd, params->lpText, params->lpCaption, params->uType);

    // Cleanup
    delete params; // Remember to free the allocated memory for parameters
    return 0;
}

int main()
{
   
    /*
    HANDLE th = OpenThread(THREAD_GET_CONTEXT  | THREAD_SUSPEND_RESUME, FALSE, 3956);

    if (!th) {
        printf("error %d", GetLastError());
    }

    printf("pass : %d", th);

    ResumeThread(th);


    return 0;
    */

    DWORD pid = GetProcessIdByName("notepad.exe");
    std::cout << " PID OF PROCESS : " << pid << std::endl;
    //HANDLE hProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid);
    HANDLE hProcess = OpenProcess(PROCESS_CREATE_THREAD | PROCESS_QUERY_INFORMATION | PROCESS_VM_OPERATION | PROCESS_VM_WRITE | PROCESS_VM_READ, FALSE, pid);

    //if (!EnableDebugPrivilege(GetCurrentProcess())) {
   //     std::cerr << "Failed to enable SeDebugPrivilege." << std::endl;
    //    return 1;
   // }


    if (hProcess == NULL) {
        std::cerr << "Failed to open process." << std::endl;
        return 1;
    }


    // The message to show in MessageBox
    const char* msg = "Injected Message!";
    SIZE_T msgSize = strlen(msg) + 1;

    LPVOID remoteMessage = VirtualAllocEx(hProcess, NULL, msgSize, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);

    if (remoteMessage == NULL) {
        std::cerr << "Failed to allocate memory in target process." << std::endl;
        CloseHandle(hProcess);
        return 1;
    }

    /*
    // Write the message to the allocated memory in the remote process
    //WriteProcessMemory(hProcess, remoteMessage, msg, msgSize, NULL);

      // Allocate memory in the target process for our custom thread function
    void* remoteFuncMemory = VirtualAllocEx(hProcess, NULL, sizeof(customThreadFunc), MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
    if (!remoteFuncMemory) {
        std::cerr << "VirtualAllocEx failed: " << GetLastError() << std::endl;
        CloseHandle(hProcess);
        return false;
    }

    // Write the custom thread function code to the allocated memory
    if (!WriteProcessMemory(hProcess, remoteFuncMemory, customThreadFunc, sizeof(customThreadFunc), NULL)) {
        std::cerr << "WriteProcessMemory failed: " << GetLastError() << std::endl;
        VirtualFreeEx(hProcess, remoteFuncMemory, 0, MEM_RELEASE);
        CloseHandle(hProcess);
        return false;
    }



    // 
    HMODULE hUser321 = GetModuleHandle(L"user32.dll");
    FARPROC pMessageBox1 = GetProcAddress(hUser321, "MessageBoxA");

    printf("mbox %d %d  \n", pMessageBox1, hUser321);

    HMODULE hUser32 = LoadLibrary(L"c:\\Windows\\System32\\user32.dll");  // Load user32.dll (or get handle if it's already loaded)
    if (hUser32 == NULL) {
        // Handle error (for example, library not found)
        return 1;
    }

    FARPROC pMessageBox = GetProcAddress(hUser32, "MessageBoxA");
    if (pMessageBox == NULL) {
        // Handle error (for example, function not found in the library)
        FreeLibrary(hUser32);  // Cleanup if necessary
        return 1;
    }

    //pMessageBox(NULL, L"test", L"xxx", MB_OK);

    // Cast the function pointer to the correct type
    typedef int (WINAPI* MessageBoxA_t)(HWND, LPCSTR, LPCSTR, UINT);
    MessageBoxA_t msgBox = (MessageBoxA_t)pMessageBox;

    // Call the MessageBoxA function
    msgBox(NULL, "Test Message", "Test Title", MB_OK);

    printf("%d %d \n", pMessageBox, remoteMessage);
    /*
    *

    HANDLE hThread = CreateRemoteThread(
        hProcess,
        NULL,
        0,
        (LPTHREAD_START_ROUTINE)pMessageBox,
        remoteMessage,
        0,
        NULL
    );

    if (hThread == NULL) {
        std::cerr << "Failed to create remote thread." << GetLastError() << std::endl;
    }
    else {
        std::cout << "Remote thread created successfully!" << std::endl;
    }

    // Wait for the remote thread to complete
    WaitForSingleObject(hThread, INFINITE);

    // Clean up
    VirtualFreeEx(hProcess, remoteMessage, 0, MEM_RELEASE);
    CloseHandle(hThread);
    CloseHandle(hProcess);
    */

    /*
    

    // Create the remote thread with our custom function
    DWORD is = NULL;
    HANDLE hThread = CreateRemoteThreadEx(hProcess, NULL, 0, (LPTHREAD_START_ROUTINE)pMessageBox, NULL, CREATE_SUSPENDED, NULL, &is);
    if (!hThread) {
        std::cerr << "CreateRemoteThread failed: " << GetLastError() << std::endl;
        VirtualFreeEx(hProcess, remoteFuncMemory, 0, MEM_RELEASE);
        CloseHandle(hProcess);
        return false;
    }

    // Wait for the remote thread to execute (optional)
    WaitForSingleObject(hThread, INFINITE);

    // Cleanup
    VirtualFreeEx(hProcess, remoteFuncMemory, 0, MEM_RELEASE);
    CloseHandle(hThread);
    CloseHandle(hProcess);
    */
HMODULE hUser321 = GetModuleHandle(L"user32.dll");
FARPROC pMessageBox1 = GetProcAddress(hUser321, "MessageBoxA");

printf("mbox %d %d  \n", pMessageBox1, hUser321);

HMODULE hUser32 = LoadLibrary(L"c:\\Windows\\System32\\user32.dll");  // Load user32.dll (or get handle if it's already loaded)
if (hUser32 == NULL) {
    // Handle error (for example, library not found)
    return 1;
}

FARPROC pMessageBox = GetProcAddress(hUser32, "MessageBoxA");
if (pMessageBox == NULL) {
    // Handle error (for example, function not found in the library)
    FreeLibrary(hUser32);  // Cleanup if necessary
    return 1;
}



// Write the custom thread routine to the allocated memory
// NOTE: You must replace this with actual shellcode for CustomThreadRoutine
// For example, you could compile this function to a binary blob.

// Create the remote thread
HANDLE hThread = CreateRemoteThread(hProcess, NULL, 0, (LPTHREAD_START_ROUTINE)pMessageBox, NULL, CREATE_SUSPENDED, NULL);
if (!hThread) {
    std::cerr << "CreateRemoteThreadEx failed: " << GetLastError() << std::endl;
    
    CloseHandle(hProcess);
    return false;
}

// Optionally resume the thread if needed
//ResumeThread(hThread);

// Wait for the thread to finish if needed
WaitForSingleObject(hThread, INFINITE);

// Cleanup
CloseHandle(hThread);
CloseHandle(hProcess);
    


    //FreeLibrary(hUser32);  // Unload user32.dll

    return 0;
}