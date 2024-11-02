#include <windows.h>
#include <stdio.h>

#define SHARED_MEMORY_NAME "Local\\MySharedMemory"
#define SHARED_MEMORY_SIZE 256

DWORD WINAPI MonitorStack(LPVOID lpParam) {
    // Get the handle of the main thread
    HANDLE hThread = (HANDLE)lpParam;

    // Create a shared memory block
    HANDLE hMapFile = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, SHARED_MEMORY_SIZE, SHARED_MEMORY_NAME);
    if (hMapFile == NULL) {
        printf("Could not create file mapping object (Error %d)\n", GetLastError());
        return 1;
    }

    // Map a view of the shared memory
    BYTE* pBuf = (BYTE*)MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, SHARED_MEMORY_SIZE);
    if (pBuf == NULL) {
        printf("Could not map view of file (Error %d)\n", GetLastError());
        CloseHandle(hMapFile);
        return 1;
    }

    CONTEXT context;
    context.ContextFlags = CONTEXT_FULL;

    printf("Writer Process: Monitoring and writing stack data to shared memory...\n");

    // Continuously get the stack data
    while (1) {
        // Suspend the main thread to safely capture the context
        SuspendThread(hThread);

        // Capture the context of the main thread, including stack pointer (ESP/RSP)
        if (GetThreadContext(hThread, &context)) {
#ifdef _M_IX86
            DWORD esp = context.Esp;  // Stack pointer (ESP) on x86
#else
            DWORD64 rsp = context.Rsp; // Stack pointer (RSP) on x64
#endif
            // Copy the stack data starting from the stack pointer
            // Note: This copies a maximum of SHARED_MEMORY_SIZE bytes
            CopyMemory(pBuf, (LPCVOID)esp, SHARED_MEMORY_SIZE);
			
			printf("state : %d %d %x %x \n", context.Eax, context.Ebx, context.Eax, context.Ebx);

            printf("Writer: Updated shared memory with current stack snapshot\n");
        } else {
            printf("Could not get thread context (Error %d)\n", GetLastError());
        }

        // Resume the main thread after capturing the stack
        ResumeThread(hThread);

        Sleep(1000);  // Update every 1 second
    }

    // Cleanup (unreachable in this example, as it runs indefinitely)
    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);
    return 0;
}

DWORD WINAPI InfiniteLoop(LPVOID lpParam) {
	static int x = 0;
    __asm {
		mov eax, 0xdeadc0de
    loop_start:
        //
        //inc eax
		push eax
		pop ebx
        jmp loop_start
    }
    return 0;
}

int main() {
    // Start the main infinite loop in a separate thread
    DWORD threadId;
    HANDLE hMainThread = CreateThread(NULL, 0, InfiniteLoop, NULL, 0, &threadId);

    if (hMainThread == NULL) {
        printf("Could not create thread (Error %d)\n", GetLastError());
        return 1;
    }

    // Start the monitor thread to capture the stack of the main thread
    MonitorStack(hMainThread);

    // Wait indefinitely (the MonitorStack function runs an infinite loop)
    WaitForSingleObject(hMainThread, INFINITE);

    return 0;
}
