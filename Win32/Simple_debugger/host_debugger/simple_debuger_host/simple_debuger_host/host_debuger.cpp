#if 0
#include <windows.h>
#include <stdio.h>

#define NUM_INSTRUCTIONS 5

// Helper function to read memory from the debugged process
BOOL ReadProcessMemorySafe(HANDLE hProcess, LPCVOID lpBaseAddress, LPVOID lpBuffer, SIZE_T nSize) {
    SIZE_T bytesRead;
    if (ReadProcessMemory(hProcess, lpBaseAddress, lpBuffer, nSize, &bytesRead) && bytesRead == nSize) {
        return TRUE;
    }
    return FALSE;
}

// Function to print instructions around the current instruction pointer
void PrintInstructionsAroundIP(HANDLE hProcess, LPVOID currentAddress) {
    BYTE buffer[NUM_INSTRUCTIONS * 10];  // Buffer to store 10 bytes per instruction (estimate)
    LPVOID baseAddress = (LPBYTE)currentAddress - NUM_INSTRUCTIONS * 10;  // Start reading before EIP/RIP

    if (!ReadProcessMemorySafe(hProcess, baseAddress, buffer, sizeof(buffer))) {
        printf("Failed to read process memory.\n");
        return;
    }

    printf("Instructions around current EIP/RIP:\n");
    for (int i = -NUM_INSTRUCTIONS; i <= NUM_INSTRUCTIONS; i++) {
        LPVOID instructionAddress = (LPBYTE)currentAddress + i * 10;
        printf("0x%p: ", instructionAddress);
        for (int j = 0; j < 10; j++) {
            printf("%02X ", buffer[(i + NUM_INSTRUCTIONS) * 10 + j]);
        }
        printf("\n");
    }
}

// Function to handle exceptions and implement step-into
void HandleExceptionEvent(EXCEPTION_DEBUG_INFO* pExceptionDebugInfo, HANDLE hProcess, DWORD dwThreadId) {
    DWORD exceptionCode = pExceptionDebugInfo->ExceptionRecord.ExceptionCode;
    CONTEXT context;
    context.ContextFlags = CONTEXT_FULL;
    HANDLE hThread = OpenThread(THREAD_ALL_ACCESS, FALSE, dwThreadId);

    if (!hThread) {
        printf("Failed to open thread. Error: %lu\n", GetLastError());
        return;
    }

    if (!GetThreadContext(hThread, &context)) {
        printf("Failed to get thread context. Error: %lu\n", GetLastError());
        CloseHandle(hThread);
        return;
    }

#ifdef _WIN64
    LPVOID currentInstruction = (LPVOID)(context.Rip);  // Use Rip for x64
#else
    LPVOID currentInstruction = (LPVOID)(context.Eip);  // Use Eip for x86
#endif

    printf("\nException Code: 0x%08x at address 0x%p\n", exceptionCode, currentInstruction);

    // Print instructions around the current instruction pointer
    PrintInstructionsAroundIP(hProcess, currentInstruction);

    // Handle specific exceptions
    switch (exceptionCode) {
    case EXCEPTION_ACCESS_VIOLATION:
        printf("Access Violation! Possible buffer overflow.\n");
        break;
    case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:
        printf("Array Bounds Exceeded!\n");
        break;
    case EXCEPTION_INT_DIVIDE_BY_ZERO:
        printf("Divide by Zero!\n");
        break;
    case EXCEPTION_BREAKPOINT:
        printf("Breakpoint reached.\n");
        break;
    case EXCEPTION_SINGLE_STEP:
        printf("Single Step Event.\n");
        break;
    default:
        printf("Unhandled Exception: 0x%08x\n", exceptionCode);
        break;
    }

    CloseHandle(hThread);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <PID>\n", argv[0]);
        return 1;
    }

    DWORD pid = atoi(argv[1]);

    // Attach to the target process
    if (!DebugActiveProcess(pid)) {
        printf("Failed to attach to process %d. Error: %lu\n", pid, GetLastError());
        return 1;
    }

    DEBUG_EVENT debugEvent;
    BOOL continueDebugging = TRUE;

    while (continueDebugging) {
        // Wait for a debug event
        if (WaitForDebugEvent(&debugEvent, INFINITE)) {
            DWORD continueStatus = DBG_CONTINUE;

            switch (debugEvent.dwDebugEventCode) {
            case EXCEPTION_DEBUG_EVENT: {
                HandleExceptionEvent(&debugEvent.u.Exception, OpenProcess(PROCESS_VM_READ, FALSE, pid), debugEvent.dwThreadId);
                continueStatus = DBG_CONTINUE;

                // Set the thread to single-step mode if a breakpoint or exception was hit
                HANDLE hThread = OpenThread(THREAD_ALL_ACCESS, FALSE, debugEvent.dwThreadId);
                CONTEXT context;
                context.ContextFlags = CONTEXT_CONTROL;
                GetThreadContext(hThread, &context);
#ifdef _WIN64
                context.EFlags |= 0x100;  // Set the Trap Flag (TF) for single-step
#else
                context.EFlags |= 0x100;  // Set the Trap Flag (TF) for single-step
#endif
                SetThreadContext(hThread, &context);
                CloseHandle(hThread);

                break;
            }

            case CREATE_THREAD_DEBUG_EVENT:
                printf("Thread created. Thread ID: %lu\n", debugEvent.dwThreadId);
                break;

            case CREATE_PROCESS_DEBUG_EVENT:
                printf("Process created. Process ID: %lu\n", debugEvent.dwProcessId);
                break;

            case EXIT_THREAD_DEBUG_EVENT:
                printf("Thread exited. Thread ID: %lu\n", debugEvent.dwThreadId);
                break;

            case EXIT_PROCESS_DEBUG_EVENT:
                printf("Process exited. Process ID: %lu\n", debugEvent.dwProcessId);
                continueDebugging = FALSE;
                break;

            case LOAD_DLL_DEBUG_EVENT:
                printf("DLL loaded.\n");
                break;

            case UNLOAD_DLL_DEBUG_EVENT:
                printf("DLL unloaded.\n");
                break;

            case OUTPUT_DEBUG_STRING_EVENT:
                printf("Debug string.\n");
                break;

            default:
                printf("Unknown event: %d\n", debugEvent.dwDebugEventCode);
                break;
            }

            // Continue the debugging event
            if (!ContinueDebugEvent(debugEvent.dwProcessId, debugEvent.dwThreadId, continueStatus)) {
                printf("Failed to continue debugging. Error: %lu\n", GetLastError());
                break;
            }
        }
        else {
            printf("Failed to wait for debug event. Error: %lu\n", GetLastError());
            break;
        }
    }

    return 0;
}
#endif