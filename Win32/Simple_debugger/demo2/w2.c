#include <windows.h>
#include <stdio.h>

#define SHARED_MEMORY_NAME "Local\\MySharedMemory"
#define SHARED_MEMORY_SIZE sizeof(RegisterState)  // Size for the structure

typedef struct {
    DWORD eax;
    DWORD ebx;
    DWORD ecx;
    DWORD edx;
    DWORD esi;
    DWORD edi;
    DWORD ebp;
    DWORD esp;
    DWORD eip;
} RegisterState;

DWORD WINAPI MonitorRegisters(LPVOID lpParam) {
    // Get the handle of the main thread
    HANDLE hThread = (HANDLE)lpParam;

    // Create a shared memory block
    HANDLE hMapFile = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, SHARED_MEMORY_SIZE, SHARED_MEMORY_NAME);
    if (hMapFile == NULL) {
        printf("Could not create file mapping object (Error %d)\n", GetLastError());
        return 1;
    }

    // Map a view of the shared memory
    RegisterState* pBuf = (RegisterState*)MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, SHARED_MEMORY_SIZE);
    if (pBuf == NULL) {
        printf("Could not map view of file (Error %d)\n", GetLastError());
        CloseHandle(hMapFile);
        return 1;
    }

    CONTEXT context;
    context.ContextFlags = CONTEXT_FULL;

    printf("Writer Process: Monitoring and writing register states to shared memory...\n");

    // Continuously get the register states
    while (1) {
		
		Sleep(10 * 1000);
		__asm {
			int 3
		}
		
        // Suspend the main thread to safely capture the context
        SuspendThread(hThread);

        // Capture the context of the main thread, including registers
        if (GetThreadContext(hThread, &context)) {
            // Populate the RegisterState structure with the current register values
            pBuf->eax = context.Eax;
            pBuf->ebx = context.Ebx;
            pBuf->ecx = context.Ecx;
            pBuf->edx = context.Edx;
            pBuf->esi = context.Esi;
            pBuf->edi = context.Edi;
            pBuf->ebp = context.Ebp;
            pBuf->esp = context.Esp;
            pBuf->eip = context.Eip;

            printf("Writer: Updated shared memory with current register states\n");
        } else {
            printf("Could not get thread context (Error %d)\n", GetLastError());
        }

        // Resume the main thread after capturing the registers
        ResumeThread(hThread);

        Sleep(100);  // Update every 1 second
    }

    // Cleanup (unreachable in this example, as it runs indefinitely)
    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);
    return 0;
}

/*
	// BASIC TREAHD DEMO #1
DWORD WINAPI InfiniteLoop(LPVOID lpParam) {
    __asm {
		mov eax, 0
    loop_start:
        //mov eax, 0
		mov ebx, 0xdeadc0de
        inc eax
        jmp loop_start
    }
    return 0;
}
*/


DWORD WINAPI InfiniteLoop(LPVOID lpParam) {
    // Variables to hold information for the registers
    DWORD heapStart = 0, stackStart = 0, threadId = GetCurrentThreadId(), processId = GetCurrentProcessId();
    
    // Get memory information
    MEMORY_BASIC_INFORMATION mbi;

    // Get heap base address (this may vary depending on the process and memory layout)
    HANDLE hHeap = GetProcessHeap();
    if (hHeap) {
        heapStart = (DWORD)HeapAlloc(hHeap, HEAP_ZERO_MEMORY, 0);
        if (heapStart == 0) {
            printf("Failed to get heap address\n");
        }
    }

    // Get stack base address
    if (VirtualQuery(&mbi, &mbi, sizeof(mbi))) {
        stackStart = (DWORD)mbi.BaseAddress;
    }
	
	
	// create scenario for debugger - scenario 1
	Sleep(20 * 1000); // 20 sec sleep time and wait for debugger
	
	
	static int x_val = 0;

    // Infinite loop in assembly with register assignments
    __asm {
		// if 20 sec sleep time - scenario 1
		int 3
		//////////////////////////////////////////
        mov eax, 0                // Clear EAX
        mov ebx, 0xdeadc0de      // Just a sample value for EBX
        mov ecx, threadId        // Store Thread ID in ECX
        mov edx, processId       // Store Process ID in EDX
        mov esi, heapStart        // Store Heap Start Address in ESI
        mov edi, stackStart       // Store Stack Start Address in EDI
		
		//
		push eax
		mov eax, [x_val]
		inc eax
		mov x_val, eax
		pop eax
		//

    loop_start:
        inc eax                   // Increment EAX
        jmp loop_start            // Infinite loop
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

    // Start the monitor thread to capture the register states of the main thread
    MonitorRegisters(hMainThread);

    // Wait indefinitely (the MonitorRegisters function runs an infinite loop)
    WaitForSingleObject(hMainThread, INFINITE);

    return 0;
}
