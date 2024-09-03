#include <windows.h>
#include <stdio.h>
#include <conio.h>

void PrintRegisters(CONTEXT* context) {
#ifdef _WIN64
    printf("RIP: 0x%llx\n", context->Rip);
    printf("RAX: 0x%llx\n", context->Rax);
    printf("RBX: 0x%llx\n", context->Rbx);
    printf("RCX: 0x%llx\n", context->Rcx);
    printf("RDX: 0x%llx\n", context->Rdx);
    printf("RSI: 0x%llx\n", context->Rsi);
    printf("RDI: 0x%llx\n", context->Rdi);
    printf("RBP: 0x%llx\n", context->Rbp);
    printf("RSP: 0x%llx\n", context->Rsp);
    printf("R8: 0x%llx\n", context->R8);
    printf("R9: 0x%llx\n", context->R9);
    printf("R10: 0x%llx\n", context->R10);
    printf("R11: 0x%llx\n", context->R11);
    printf("R12: 0x%llx\n", context->R12);
    printf("R13: 0x%llx\n", context->R13);
    printf("R14: 0x%llx\n", context->R14);
    printf("R15: 0x%llx\n", context->R15);
#else
    printf("EIP: 0x%x\n", context->Eip);
    printf("EAX: 0x%x\n", context->Eax);
    printf("EBX: 0x%x\n", context->Ebx);
    printf("ECX: 0x%x\n", context->Ecx);
    printf("EDX: 0x%x\n", context->Edx);
    printf("ESI: 0x%x\n", context->Esi);
    printf("EDI: 0x%x\n", context->Edi);
    printf("EBP: 0x%x\n", context->Ebp);
    printf("ESP: 0x%x\n", context->Esp);
#endif
}

void SingleStepDebugger(HANDLE hProcess, DWORD dwThreadId) {
    CONTEXT context;
    HANDLE hThread = OpenThread(THREAD_ALL_ACCESS, FALSE, dwThreadId);

    if (!hThread) {
        printf("Failed to open thread. Error: %lu\n", GetLastError());
        return;
    }

    // Enter stepping loop
    while (1) {
        printf("Press 's' to step into the next instruction, 'p' to print registers, or 'q' to quit: ");
        char command = getchar();
        getchar();  // Consume newline

        if (command == 'p') {
            context.ContextFlags = CONTEXT_FULL;
            if (GetThreadContext(hThread, &context)) {
                PrintRegisters(&context);
            }
            else {
                printf("Failed to get thread context. Error: %lu\n", GetLastError());
            }
        }
        else if (command == 's') {
            // Get current thread context
            context.ContextFlags = CONTEXT_CONTROL;
            if (!GetThreadContext(hThread, &context)) {
                printf("Failed to get thread context. Error: %lu\n", GetLastError());
                break;
            }

            // Set the Trap Flag (TF) for single-step mode
#ifdef _WIN64
            context.EFlags |= 0x100;
#else
            context.EFlags |= 0x100;
#endif
            if (!SetThreadContext(hThread, &context)) {
                printf("Failed to set thread context. Error: %lu\n", GetLastError());
                break;
            }

            // Continue the process to execute one instruction
            if (!ContinueDebugEvent(GetProcessId(hProcess), dwThreadId, DBG_CONTINUE)) {
                printf("Failed to continue debugging. Error: %lu\n", GetLastError());
                break;
            }

            // Wait for the single-step exception
            DEBUG_EVENT debugEvent;
            if (WaitForDebugEvent(&debugEvent, INFINITE)) {
                if (debugEvent.dwDebugEventCode == EXCEPTION_DEBUG_EVENT &&
                    debugEvent.u.Exception.ExceptionRecord.ExceptionCode == EXCEPTION_SINGLE_STEP) {
                    printf("Single-step exception at 0x%p\n", (void*)context.Eip);
                }
                else {
                    printf("Unexpected debug event. Code: %lu\n", debugEvent.dwDebugEventCode);
                }
            }
            else {
                printf("Failed to wait for debug event. Error: %lu\n", GetLastError());
                break;
            }
        }
        else if (command == 'q') {
            break;
        }
        else {
            printf("Unknown command.\n");
        }
    }

    CloseHandle(hThread);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <PID>\n", argv[0]);
        return 1;
    }

    DWORD pid = atoi(argv[1]);
    HANDLE hProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid);
    if (!hProcess) {
        printf("Failed to open process. Error: %lu\n", GetLastError());
        return 1;
    }

    // Attach to the process
    if (!DebugActiveProcess(pid)) {
        printf("Failed to attach to process. Error: %lu\n", GetLastError());
        return 1;
    }

    BOOL continueDebugging = TRUE;
    DEBUG_EVENT debugEvent;
    DWORD continueStatus = DBG_CONTINUE;

    while (continueDebugging) {
        if (WaitForDebugEvent(&debugEvent, INFINITE)) {
            switch (debugEvent.dwDebugEventCode) {
            case EXCEPTION_DEBUG_EVENT:
                printf("Exception caught in process. Exception code: 0x%08lx\n", debugEvent.u.Exception.ExceptionRecord.ExceptionCode);
                SingleStepDebugger(hProcess, debugEvent.dwThreadId);
                break;

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

            // Continue to wait for events unless the process is exiting
            if (continueDebugging) {
                ContinueDebugEvent(debugEvent.dwProcessId, debugEvent.dwThreadId, continueStatus);
            }
        }
        else {
            printf("Failed to wait for debug event. Error: %lu\n", GetLastError());
            break;
        }
    }

    return 0;
}








#if 0 
#include <windows.h>
#include <stdio.h>
#include <conio.h>

#define NUM_INSTRUCTIONS 5

typedef struct {
    BYTE opcode;
    const char* mnemonic;
} OpcodeMnemonic;

OpcodeMnemonic simpleOpcodeTable[] = {
    { 0x90, "NOP" },
    { 0xCC, "INT 3" },
    { 0xC3, "RET" },
    { 0x55, "PUSH EBP" },
    { 0x89, "MOV" },
    { 0xE9, "JMP" },
    { 0xEB, "JMP SHORT" },
    { 0x74, "JZ" },
    { 0x75, "JNZ" },
    { 0x50, "PUSH EAX" },
    { 0x58, "POP EAX" },
    { 0xB8, "MOV EAX," },
    { 0xB9, "MOV ECX," },
    // Add more opcodes as needed...
};

BOOL ReadProcessMemorySafe(HANDLE hProcess, LPCVOID lpBaseAddress, LPVOID lpBuffer, SIZE_T nSize) {
    SIZE_T bytesRead;
    return ReadProcessMemory(hProcess, lpBaseAddress, lpBuffer, nSize, &bytesRead) && bytesRead == nSize;
}

void DisassembleInstruction(HANDLE hProcess, LPVOID instructionAddress) {
    BYTE opcode;
    if (!ReadProcessMemorySafe(hProcess, instructionAddress, &opcode, sizeof(BYTE))) {
        printf("Failed to read instruction at 0x%p\n", instructionAddress);
        return;
    }

    const char* mnemonic = "UNKNOWN";
    for (int i = 0; i < sizeof(simpleOpcodeTable) / sizeof(OpcodeMnemonic); i++) {
        if (simpleOpcodeTable[i].opcode == opcode) {
            mnemonic = simpleOpcodeTable[i].mnemonic;
            break;
        }
    }

    printf("0x%p: %02X %s\n", instructionAddress, opcode, mnemonic);
}

void PrintInstructionsAroundIP(HANDLE hProcess, LPVOID currentAddress) {
    BYTE buffer[NUM_INSTRUCTIONS * 10];
    LPVOID baseAddress = (LPBYTE)currentAddress - NUM_INSTRUCTIONS * 10;

    if (!ReadProcessMemorySafe(hProcess, baseAddress, buffer, sizeof(buffer))) {
        printf("Failed to read process memory.\n");
        return;
    }

    printf("Instructions around current EIP/RIP:\n");
    for (int i = -NUM_INSTRUCTIONS; i <= NUM_INSTRUCTIONS; i++) {
        LPVOID instructionAddress = (LPBYTE)currentAddress + i;
        printf("0x%p: ", instructionAddress);
        DisassembleInstruction(hProcess, instructionAddress);
    }
}

void PrintRegisterState(CONTEXT* context) {
#ifdef _WIN64
    printf("RIP: 0x%llx\n", context->Rip);
    printf("RAX: 0x%llx\n", context->Rax);
    printf("RBX: 0x%llx\n", context->Rbx);
    printf("RCX: 0x%llx\n", context->Rcx);
    printf("RDX: 0x%llx\n", context->Rdx);
    printf("RSI: 0x%llx\n", context->Rsi);
    printf("RDI: 0x%llx\n", context->Rdi);
    printf("RBP: 0x%llx\n", context->Rbp);
    printf("RSP: 0x%llx\n", context->Rsp);
    printf("R8: 0x%llx\n", context->R8);
    printf("R9: 0x%llx\n", context->R9);
    printf("R10: 0x%llx\n", context->R10);
    printf("R11: 0x%llx\n", context->R11);
    printf("R12: 0x%llx\n", context->R12);
    printf("R13: 0x%llx\n", context->R13);
    printf("R14: 0x%llx\n", context->R14);
    printf("R15: 0x%llx\n", context->R15);
#else
    printf("EIP: 0x%x\n", context->Eip);
    printf("EAX: 0x%x\n", context->Eax);
    printf("EBX: 0x%x\n", context->Ebx);
    printf("ECX: 0x%x\n", context->Ecx);
    printf("EDX: 0x%x\n", context->Edx);
    printf("ESI: 0x%x\n", context->Esi);
    printf("EDI: 0x%x\n", context->Edi);
    printf("EBP: 0x%x\n", context->Ebp);
    printf("ESP: 0x%x\n", context->Esp);
#endif
}

void StepIntoInstruction(HANDLE hProcess, HANDLE hThread, CONTEXT* context) {
    context->ContextFlags = CONTEXT_CONTROL;
    if (!GetThreadContext(hThread, context)) {
        printf("Failed to get thread context. Error: %lu\n", GetLastError());
        return;
    }

#ifdef _WIN64
    context->EFlags |= 0x100;  // Set the Trap Flag (TF) for single-step
#else
    context->EFlags |= 0x100;  // Set the Trap Flag (TF) for single-step
#endif

    if (!SetThreadContext(hThread, context)) {
        printf("Failed to set thread context. Error: %lu\n", GetLastError());
    }
}

void ContinueExecutionAfterStep(HANDLE hProcess, HANDLE hThread, CONTEXT* context) {
    // Continue execution after single-stepping
    if (!ContinueDebugEvent(GetProcessId(hProcess), GetThreadId(hThread), DBG_CONTINUE)) {
        printf("Failed to continue debugging. Error: %lu\n", GetLastError());
    }
}

void SingleStepDebugger(HANDLE hProcess, DWORD dwThreadId) {
    CONTEXT context;
    HANDLE hThread = OpenThread(THREAD_ALL_ACCESS, FALSE, dwThreadId);

    if (!hThread) {
        printf("Failed to open thread. Error: %lu\n", GetLastError());
        return;
    }

    context.ContextFlags = CONTEXT_CONTROL;

    while (1) {
        // Get the current thread context to get the instruction pointer
        if (!GetThreadContext(hThread, &context)) {
            printf("Failed to get thread context. Error: %lu\n", GetLastError());
            break;
        }

        // Disassemble current instruction
        PrintInstructionsAroundIP(hProcess, (LPVOID)context.Eip);

        // Wait for user command
        printf("Press 's' to step into the next instruction, 'p' to print registers, or 'c' to continue: ");
        char command = getchar();
        getchar();  // Consume newline

        if (command == 'p') {
            context.ContextFlags = CONTEXT_FULL;
            if (GetThreadContext(hThread, &context)) {
                PrintRegisterState(&context);
            }
            else {
                printf("Failed to get thread context. Error: %lu\n", GetLastError());
            }
        }
        else if (command == 's') {
            StepIntoInstruction(hProcess, hThread, &context);
            // Continue execution for a single step
            ContinueExecutionAfterStep(hProcess, hThread, &context);

            // Wait for the single-step exception
            DEBUG_EVENT debugEvent;
            if (WaitForDebugEvent(&debugEvent, INFINITE)) {
                // Process the single-step exception
                if (debugEvent.dwDebugEventCode == EXCEPTION_DEBUG_EVENT &&
                    debugEvent.u.Exception.ExceptionRecord.ExceptionCode == EXCEPTION_SINGLE_STEP) {
                    printf("Single-step exception occurred.\n");
                }
            }
            else {
                printf("Failed to wait for debug event. Error: %lu\n", GetLastError());
                break;
            }
        }
        else if (command == 'c') {
            break;
        }
        else {
            printf("Unknown command.\n");
        }
    }

    CloseHandle(hThread);
}

void DetectCtrlDAndBreakpoint(HANDLE hProcess, HANDLE hThread) {
    // Wait for CTRL+D signal or equivalent
    CONTEXT context;
    while (1) {
        char ch = getchar();
        if (ch == 4) {  // CTRL+D is ASCII code 4
            printf("CTRL+D detected. Entering single-step mode.\n");

            // Get the current thread context
            context.ContextFlags = CONTEXT_FULL;
            if (!GetThreadContext(hThread, &context)) {
                printf("Failed to get thread context. Error: %lu\n", GetLastError());
                return;
            }

            PrintRegisterState(&context);
            StepIntoInstruction(hProcess, hThread, &context);

            // Continue execution for a single step
            ContinueExecutionAfterStep(hProcess, hThread, &context);
            break;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <PID>\n", argv[0]);
        return 1;
    }

    DWORD pid = atoi(argv[1]);
    HANDLE hProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid);
    if (!hProcess) {
        printf("Failed to open process. Error: %lu\n", GetLastError());
        return 1;
    }

    BOOL continueDebugging = TRUE;
    DEBUG_EVENT debugEvent;
    DWORD continueStatus = DBG_CONTINUE;
    BOOL stepInto = FALSE;

    // Attach to the process
    if (!DebugActiveProcess(pid)) {
        printf("Failed to attach to process. Error: %lu\n", GetLastError());
        return 1;
    }

    HANDLE hThread = NULL;

    while (continueDebugging) {
        if (WaitForDebugEvent(&debugEvent, INFINITE)) {
            switch (debugEvent.dwDebugEventCode) {
            case EXCEPTION_DEBUG_EVENT:
                printf("Exception caught in process. Exception code: 0x%08lx\n", debugEvent.u.Exception.ExceptionRecord.ExceptionCode);
                stepInto = TRUE;
                break;

            case CREATE_THREAD_DEBUG_EVENT:
                printf("Thread created. Thread ID: %lu\n", debugEvent.dwThreadId);
                hThread = OpenThread(THREAD_ALL_ACCESS, FALSE, debugEvent.dwThreadId);
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

            if (stepInto) {
                SingleStepDebugger(hProcess, debugEvent.dwThreadId);
                stepInto = FALSE;  // Reset step-into flag
            }

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

    DetectCtrlDAndBreakpoint(hProcess, hThread);

    return 0;
}
#endif






#if 0
#include <windows.h>
#include <stdio.h>

#define NUM_INSTRUCTIONS 5

typedef struct {
    BYTE opcode;
    const char* mnemonic;
} OpcodeMnemonic;

OpcodeMnemonic simpleOpcodeTable[] = {
    { 0x90, "NOP" },
    { 0xCC, "INT 3" },
    { 0xC3, "RET" },
    { 0x55, "PUSH EBP" },
    { 0x89, "MOV" },
    { 0xE9, "JMP" },
    { 0xEB, "JMP SHORT" },
    { 0x74, "JZ" },
    { 0x75, "JNZ" },
    { 0x50, "PUSH EAX" },
    { 0x58, "POP EAX" },
    { 0xB8, "MOV EAX," },
    { 0xB9, "MOV ECX," },
    // Add more opcodes as needed...
};

BOOL ReadProcessMemorySafe(HANDLE hProcess, LPCVOID lpBaseAddress, LPVOID lpBuffer, SIZE_T nSize) {
    SIZE_T bytesRead;
    return ReadProcessMemory(hProcess, lpBaseAddress, lpBuffer, nSize, &bytesRead) && bytesRead == nSize;
}

void DisassembleInstruction(HANDLE hProcess, LPVOID instructionAddress) {
    BYTE opcode;
    if (!ReadProcessMemorySafe(hProcess, instructionAddress, &opcode, sizeof(BYTE))) {
        printf("Failed to read instruction at 0x%p\n", instructionAddress);
        return;
    }

    const char* mnemonic = "UNKNOWN";
    for (int i = 0; i < sizeof(simpleOpcodeTable) / sizeof(OpcodeMnemonic); i++) {
        if (simpleOpcodeTable[i].opcode == opcode) {
            mnemonic = simpleOpcodeTable[i].mnemonic;
            break;
        }
    }

    printf("0x%p: %02X %s\n", instructionAddress, opcode, mnemonic);
}

void PrintInstructionsAroundIP(HANDLE hProcess, LPVOID currentAddress) {
    BYTE buffer[NUM_INSTRUCTIONS * 10];
    LPVOID baseAddress = (LPBYTE)currentAddress - NUM_INSTRUCTIONS * 10;

    if (!ReadProcessMemorySafe(hProcess, baseAddress, buffer, sizeof(buffer))) {
        printf("Failed to read process memory.\n");
        return;
    }

    printf("Instructions around current EIP/RIP:\n");
    for (int i = -NUM_INSTRUCTIONS; i <= NUM_INSTRUCTIONS; i++) {
        LPVOID instructionAddress = (LPBYTE)currentAddress + i;
        printf("0x%p: ", instructionAddress);
        DisassembleInstruction(hProcess, instructionAddress);
    }
}

void PrintRegisterState(CONTEXT* context) {
#ifdef _WIN64
    printf("RIP: 0x%llx\n", context->Rip);
    printf("RAX: 0x%llx\n", context->Rax);
    printf("RBX: 0x%llx\n", context->Rbx);
    printf("RCX: 0x%llx\n", context->Rcx);
    printf("RDX: 0x%llx\n", context->Rdx);
    printf("RSI: 0x%llx\n", context->Rsi);
    printf("RDI: 0x%llx\n", context->Rdi);
    printf("RBP: 0x%llx\n", context->Rbp);
    printf("RSP: 0x%llx\n", context->Rsp);
    printf("R8: 0x%llx\n", context->R8);
    printf("R9: 0x%llx\n", context->R9);
    printf("R10: 0x%llx\n", context->R10);
    printf("R11: 0x%llx\n", context->R11);
    printf("R12: 0x%llx\n", context->R12);
    printf("R13: 0x%llx\n", context->R13);
    printf("R14: 0x%llx\n", context->R14);
    printf("R15: 0x%llx\n", context->R15);
#else
    printf("EIP: 0x%x\n", context->Eip);
    printf("EAX: 0x%x\n", context->Eax);
    printf("EBX: 0x%x\n", context->Ebx);
    printf("ECX: 0x%x\n", context->Ecx);
    printf("EDX: 0x%x\n", context->Edx);
    printf("ESI: 0x%x\n", context->Esi);
    printf("EDI: 0x%x\n", context->Edi);
    printf("EBP: 0x%x\n", context->Ebp);
    printf("ESP: 0x%x\n", context->Esp);
#endif
}

void StepIntoInstruction(HANDLE hProcess, HANDLE hThread, CONTEXT* context) {
    context->ContextFlags = CONTEXT_CONTROL;
    if (!GetThreadContext(hThread, context)) {
        printf("Failed to get thread context. Error: %lu\n", GetLastError());
        return;
    }

#ifdef _WIN64
    context->EFlags |= 0x100;  // Set the Trap Flag (TF) for single-step
#else
    context->EFlags |= 0x100;  // Set the Trap Flag (TF) for single-step
#endif

    if (!SetThreadContext(hThread, context)) {
        printf("Failed to set thread context. Error: %lu\n", GetLastError());
    }
}

void ContinueExecutionAfterStep(HANDLE hProcess, HANDLE hThread, CONTEXT* context) {
    // Continue execution after single-stepping
    if (!ContinueDebugEvent(GetProcessId(hProcess), GetThreadId(hThread), DBG_CONTINUE)) {
        printf("Failed to continue debugging. Error: %lu\n", GetLastError());
    }
}

void SingleStepDebugger(HANDLE hProcess, DWORD dwThreadId) {
    CONTEXT context;
    HANDLE hThread = OpenThread(THREAD_ALL_ACCESS, FALSE, dwThreadId);

    if (!hThread) {
        printf("Failed to open thread. Error: %lu\n", GetLastError());
        return;
    }

    context.ContextFlags = CONTEXT_CONTROL;

    while (1) {
        // Get the current thread context to get the instruction pointer
        if (!GetThreadContext(hThread, &context)) {
            printf("Failed to get thread context. Error: %lu\n", GetLastError());
            break;
        }

        // Disassemble current instruction
        PrintInstructionsAroundIP(hProcess, (LPVOID)context.Eip); // Rip

        // Wait for user command
        printf("Press 's' to step into the next instruction, 'p' to print registers, or 'c' to continue: ");
        char command = getchar();
        getchar();  // Consume newline

        if (command == 'p') {
            context.ContextFlags = CONTEXT_FULL;
            if (GetThreadContext(hThread, &context)) {
                PrintRegisterState(&context);
            }
            else {
                printf("Failed to get thread context. Error: %lu\n", GetLastError());
            }
        }
        else if (command == 's') {
            StepIntoInstruction(hProcess, hThread, &context);
            // Continue execution for a single step
            ContinueExecutionAfterStep(hProcess, hThread, &context);

            // Wait for the single-step exception
            DEBUG_EVENT debugEvent;
            if (WaitForDebugEvent(&debugEvent, INFINITE)) {
                // Process the single-step exception
                if (debugEvent.dwDebugEventCode == EXCEPTION_DEBUG_EVENT &&
                    debugEvent.u.Exception.ExceptionRecord.ExceptionCode == EXCEPTION_SINGLE_STEP) {
                    printf("Single-step exception occurred.\n");
                }
            }
            else {
                printf("Failed to wait for debug event. Error: %lu\n", GetLastError());
                break;
            }
        }
        else if (command == 'c') {
            break;
        }
        else {
            printf("Unknown command.\n");
        }
    }

    CloseHandle(hThread);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <PID>\n", argv[0]);
        return 1;
    }

    DWORD pid = atoi(argv[1]);
    DEBUG_EVENT debugEvent;
    BOOL continueDebugging = TRUE;
    DWORD continueStatus = DBG_CONTINUE;
    int stepInto = 0;

    if (!DebugActiveProcess(pid)) {
        printf("Failed to attach to process. Error: %lu\n", GetLastError());
        return 1;
    }

    while (continueDebugging) {
        if (WaitForDebugEvent(&debugEvent, INFINITE)) {
            switch (debugEvent.dwDebugEventCode) {
            case EXCEPTION_DEBUG_EVENT:
                printf("Exception caught in process. Exception code: 0x%08lx\n", debugEvent.u.Exception.ExceptionRecord.ExceptionCode);
                stepInto = 1;
                break;

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

            if (stepInto) {
                SingleStepDebugger(OpenProcess(PROCESS_ALL_ACCESS, FALSE, debugEvent.dwProcessId), debugEvent.dwThreadId);
                stepInto = 0;  // Reset step-into flag
            }

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




#if 0
#include <windows.h>
#include <stdio.h>

#define NUM_INSTRUCTIONS 5

typedef struct {
    BYTE opcode;
    const char* mnemonic;
} OpcodeMnemonic;

OpcodeMnemonic simpleOpcodeTable[] = {
    { 0x90, "NOP" },
    { 0xCC, "INT 3" },
    { 0xC3, "RET" },
    { 0x55, "PUSH EBP" },
    { 0x89, "MOV" },
    { 0xE9, "JMP" },
    { 0xEB, "JMP SHORT" },
    { 0x74, "JZ" },
    { 0x75, "JNZ" },
    { 0x50, "PUSH EAX" },
    { 0x58, "POP EAX" },
    { 0xB8, "MOV EAX," },
    { 0xB9, "MOV ECX," },
    // Add more opcodes as needed...
};

void HandleExceptionEvent(EXCEPTION_DEBUG_INFO* pExceptionDebugInfo, HANDLE hProcess, DWORD dwThreadId, int* stepInto) {
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
    LPVOID currentInstruction = (LPVOID)(context.Rip);
#else
    LPVOID currentInstruction = (LPVOID)(context.Eip);
#endif

    printf("\nException Code: 0x%08x at address 0x%p\n", exceptionCode, currentInstruction);

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

    *stepInto = 1;

    CloseHandle(hThread);
}

BOOL ReadProcessMemorySafe(HANDLE hProcess, LPCVOID lpBaseAddress, LPVOID lpBuffer, SIZE_T nSize) {
    SIZE_T bytesRead;
    return ReadProcessMemory(hProcess, lpBaseAddress, lpBuffer, nSize, &bytesRead) && bytesRead == nSize;
}

void DisassembleInstruction(HANDLE hProcess, LPVOID instructionAddress) {
    BYTE opcode;
    if (!ReadProcessMemorySafe(hProcess, instructionAddress, &opcode, sizeof(BYTE))) {
        printf("Failed to read instruction at 0x%p\n", instructionAddress);
        return;
    }

    const char* mnemonic = "UNKNOWN";
    for (int i = 0; i < sizeof(simpleOpcodeTable) / sizeof(OpcodeMnemonic); i++) {
        if (simpleOpcodeTable[i].opcode == opcode) {
            mnemonic = simpleOpcodeTable[i].mnemonic;
            break;
        }
    }

    printf("0x%p: %02X %s\n", instructionAddress, opcode, mnemonic);
}

void PrintInstructionsAroundIP(HANDLE hProcess, LPVOID currentAddress) {
    BYTE buffer[NUM_INSTRUCTIONS * 10];
    LPVOID baseAddress = (LPBYTE)currentAddress - NUM_INSTRUCTIONS * 10;

    if (!ReadProcessMemorySafe(hProcess, baseAddress, buffer, sizeof(buffer))) {
        printf("Failed to read process memory.\n");
        return;
    }

    printf("Instructions around current EIP/RIP:\n");
    for (int i = -NUM_INSTRUCTIONS; i <= NUM_INSTRUCTIONS; i++) {
        LPVOID instructionAddress = (LPBYTE)currentAddress + i;
        printf("0x%p: ", instructionAddress);
        DisassembleInstruction(hProcess, instructionAddress);
    }
}

void PrintRegisterState(CONTEXT* context) {
#ifdef _WIN64
    printf("RIP: 0x%llx\n", context->Rip);
    printf("RAX: 0x%llx\n", context->Rax);
    printf("RBX: 0x%llx\n", context->Rbx);
    printf("RCX: 0x%llx\n", context->Rcx);
    printf("RDX: 0x%llx\n", context->Rdx);
    printf("RSI: 0x%llx\n", context->Rsi);
    printf("RDI: 0x%llx\n", context->Rdi);
    printf("RBP: 0x%llx\n", context->Rbp);
    printf("RSP: 0x%llx\n", context->Rsp);
    printf("R8: 0x%llx\n", context->R8);
    printf("R9: 0x%llx\n", context->R9);
    printf("R10: 0x%llx\n", context->R10);
    printf("R11: 0x%llx\n", context->R11);
    printf("R12: 0x%llx\n", context->R12);
    printf("R13: 0x%llx\n", context->R13);
    printf("R14: 0x%llx\n", context->R14);
    printf("R15: 0x%llx\n", context->R15);
#else
    printf("EIP: 0x%x\n", context->Eip);
    printf("EAX: 0x%x\n", context->Eax);
    printf("EBX: 0x%x\n", context->Ebx);
    printf("ECX: 0x%x\n", context->Ecx);
    printf("EDX: 0x%x\n", context->Edx);
    printf("ESI: 0x%x\n", context->Esi);
    printf("EDI: 0x%x\n", context->Edi);
    printf("EBP: 0x%x\n", context->Ebp);
    printf("ESP: 0x%x\n", context->Esp);
#endif
}

void StepIntoInstruction(HANDLE hProcess, HANDLE hThread, CONTEXT* context) {
    context->ContextFlags = CONTEXT_CONTROL;
    GetThreadContext(hThread, context);
#ifdef _WIN64
    context->EFlags |= 0x100;  // Set the Trap Flag (TF) for single-step
#else
    context->EFlags |= 0x100;  // Set the Trap Flag (TF) for single-step
#endif

    if (!SetThreadContext(hThread, context)) {
        printf("Failed to set thread context. Error: %lu\n", GetLastError());
    }
}

void SingleStepDebugger(HANDLE hProcess, DWORD dwThreadId) {
    CONTEXT context;
    HANDLE hThread = OpenThread(THREAD_ALL_ACCESS, FALSE, dwThreadId);

    if (!hThread) {
        printf("Failed to open thread. Error: %lu\n", GetLastError());
        return;
    }

    context.ContextFlags = CONTEXT_CONTROL;
    DEBUG_EVENT debugEvent;

    while (1) {
        if (!GetThreadContext(hThread, &context)) {
            printf("Failed to get thread context. Error: %lu\n", GetLastError());
            break;
        }

        // Wait for the user to press a key to step to the next instruction
        printf("Press 's' to step into the next instruction, 'p' to print registers, or 'c' to continue: ");
        char command = getchar();
        getchar();  // Consume newline

        if (command == 'p') {
            context.ContextFlags = CONTEXT_FULL;
            if (GetThreadContext(hThread, &context)) {
                PrintRegisterState(&context);
            }
            else {
                printf("Failed to get thread context. Error: %lu\n", GetLastError());
            }
        }
        else if (command == 's') {
            StepIntoInstruction(hProcess, hThread, &context);
            if (!ContinueDebugEvent(GetProcessId(hProcess), dwThreadId, DBG_CONTINUE)) {
                printf("Failed to continue debugging. Error: %lu\n", GetLastError());
                break;
            }
            // Wait for the single-step exception
            if (WaitForDebugEvent(&debugEvent, INFINITE)) {
                // Process the single-step exception
                if (debugEvent.dwDebugEventCode == EXCEPTION_DEBUG_EVENT &&
                    debugEvent.u.Exception.ExceptionRecord.ExceptionCode == EXCEPTION_SINGLE_STEP) {
                    printf("Single-step exception occurred.\n");
                }
            }
            else {
                printf("Failed to wait for debug event. Error: %lu\n", GetLastError());
                break;
            }
        }
        else if (command == 'c') {
            break;
        }
        else {
            printf("Unknown command.\n");
        }
    }

    CloseHandle(hThread);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <PID>\n", argv[0]);
        return 1;
    }

    DWORD pid = atoi(argv[1]);
    DEBUG_EVENT debugEvent;
    BOOL continueDebugging = TRUE;
    DWORD continueStatus = DBG_CONTINUE;
    int stepInto = 0;

    if (!DebugActiveProcess(pid)) {
        printf("Failed to attach to process %d. Error: %lu\n", pid, GetLastError());
        return 1;
    }

    while (continueDebugging) {
        if (WaitForDebugEvent(&debugEvent, INFINITE)) {
            switch (debugEvent.dwDebugEventCode) {
            case EXCEPTION_DEBUG_EVENT:
                HandleExceptionEvent(&debugEvent.u.Exception, OpenProcess(PROCESS_ALL_ACCESS, FALSE, debugEvent.dwProcessId), debugEvent.dwThreadId, &stepInto);
                if (stepInto) {
                    SingleStepDebugger(OpenProcess(PROCESS_ALL_ACCESS, FALSE, debugEvent.dwProcessId), debugEvent.dwThreadId);
                    stepInto = 0;  // Reset step-into flag
                }
                break;

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






#if 0
#include <windows.h>
#include <stdio.h>

void SingleStepDebugger(HANDLE hProcess, DWORD dwThreadId);

void HandleExceptionEvent(EXCEPTION_DEBUG_INFO* pExceptionDebugInfo, HANDLE hProcess, DWORD dwThreadId, int* stepInto) {
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
    LPVOID currentInstruction = (LPVOID)(context.Rip);
#else
    LPVOID currentInstruction = (LPVOID)(context.Eip);
#endif

    printf("\nException Code: 0x%08x at address 0x%p\n", exceptionCode, currentInstruction);

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

    *stepInto = 1;

    CloseHandle(hThread);
}

void SingleStepDebugger(HANDLE hProcess, DWORD dwThreadId) {
    CONTEXT context;
    HANDLE hThread = OpenThread(THREAD_ALL_ACCESS, FALSE, dwThreadId);

    if (!hThread) {
        printf("Failed to open thread. Error: %lu\n", GetLastError());
        return;
    }

    context.ContextFlags = CONTEXT_CONTROL;

    while (1) {
        if (!GetThreadContext(hThread, &context)) {
            printf("Failed to get thread context. Error: %lu\n", GetLastError());
            break;
        }

#ifdef _WIN64
        context.EFlags |= 0x100;  // Set the Trap Flag (TF) for single-step
#else
        context.EFlags |= 0x100;  // Set the Trap Flag (TF) for single-step
#endif

        if (!SetThreadContext(hThread, &context)) {
            printf("Failed to set thread context. Error: %lu\n", GetLastError());
            break;
        }

        printf("Press 's' to step into the next instruction, 'p' to print registers, or 'c' to continue: ");
        char command = getchar();
        getchar();  // Consume newline

        if (command == 'p') {
            context.ContextFlags = CONTEXT_FULL;
            if (GetThreadContext(hThread, &context)) {
                // Print the register state here
#ifdef _WIN64
                printf("RIP: 0x%llx\n", context.Rip);
                printf("RAX: 0x%llx\n", context.Rax);
                printf("RBX: 0x%llx\n", context.Rbx);
                // Print other registers as needed
#else
                printf("EIP: 0x%x\n", context.Eip);
                printf("EAX: 0x%x\n", context.Eax);
                printf("EBX: 0x%x\n", context.Ebx);
                // Print other registers as needed
#endif
            }
            else {
                printf("Failed to get thread context. Error: %lu\n", GetLastError());
            }
        }
        else if (command == 'c') {
            break;
        }

        if (!ContinueDebugEvent(GetProcessId(hProcess), dwThreadId, DBG_CONTINUE)) {
            printf("Failed to continue debugging. Error: %lu\n", GetLastError());
            break;
        }
    }

    CloseHandle(hThread);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <PID>\n", argv[0]);
        return 1;
    }

    DWORD pid = atoi(argv[1]);
    DEBUG_EVENT debugEvent;
    BOOL continueDebugging = TRUE;
    DWORD continueStatus = DBG_CONTINUE;
    int stepInto = 0;

    if (!DebugActiveProcess(pid)) {
        printf("Failed to attach to process %d. Error: %lu\n", pid, GetLastError());
        return 1;
    }

    while (continueDebugging) {
        if (WaitForDebugEvent(&debugEvent, INFINITE)) {
            switch (debugEvent.dwDebugEventCode) {
            case EXCEPTION_DEBUG_EVENT:
                HandleExceptionEvent(&debugEvent.u.Exception, OpenProcess(PROCESS_ALL_ACCESS, FALSE, debugEvent.dwProcessId), debugEvent.dwThreadId, &stepInto);
                if (stepInto) {
                    SingleStepDebugger(OpenProcess(PROCESS_ALL_ACCESS, FALSE, debugEvent.dwProcessId), debugEvent.dwThreadId);
                    stepInto = 0;  // Reset step-into flag
                }
                break;

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




#if 0 
#include <windows.h>
#include <stdio.h>

#define NUM_INSTRUCTIONS 5

typedef struct {
    BYTE opcode;
    const char* mnemonic;
} OpcodeMnemonic;

OpcodeMnemonic simpleOpcodeTable[] = {
    { 0x90, "NOP" },
    { 0xCC, "INT 3" },
    { 0xC3, "RET" },
    { 0x55, "PUSH EBP" },
    { 0x89, "MOV" },
    { 0xE9, "JMP" },
    { 0xEB, "JMP SHORT" },
    { 0x74, "JZ" },
    { 0x75, "JNZ" },
    { 0x50, "PUSH EAX" },
    { 0x58, "POP EAX" },
    { 0xB8, "MOV EAX," },
    { 0xB9, "MOV ECX," },
    // Add more opcodes as needed...
};

BOOL ReadProcessMemorySafe(HANDLE hProcess, LPCVOID lpBaseAddress, LPVOID lpBuffer, SIZE_T nSize) {
    SIZE_T bytesRead;
    if (ReadProcessMemory(hProcess, lpBaseAddress, lpBuffer, nSize, &bytesRead) && bytesRead == nSize) {
        return TRUE;
    }
    return FALSE;
}

void DisassembleInstruction(HANDLE hProcess, LPVOID instructionAddress) {
    BYTE opcode;
    if (!ReadProcessMemorySafe(hProcess, instructionAddress, &opcode, sizeof(BYTE))) {
        printf("Failed to read instruction at 0x%p\n", instructionAddress);
        return;
    }

    const char* mnemonic = "UNKNOWN";
    for (int i = 0; i < sizeof(simpleOpcodeTable) / sizeof(OpcodeMnemonic); i++) {
        if (simpleOpcodeTable[i].opcode == opcode) {
            mnemonic = simpleOpcodeTable[i].mnemonic;
            break;
        }
    }

    printf("0x%p: %02X %s\n", instructionAddress, opcode, mnemonic);
}

void PrintInstructionsAroundIP(HANDLE hProcess, LPVOID currentAddress) {
    BYTE buffer[NUM_INSTRUCTIONS * 10];
    LPVOID baseAddress = (LPBYTE)currentAddress - NUM_INSTRUCTIONS * 10;

    if (!ReadProcessMemorySafe(hProcess, baseAddress, buffer, sizeof(buffer))) {
        printf("Failed to read process memory.\n");
        return;
    }

    printf("Instructions around current EIP/RIP:\n");
    for (int i = -NUM_INSTRUCTIONS; i <= NUM_INSTRUCTIONS; i++) {
        LPVOID instructionAddress = (LPBYTE)currentAddress + i * 10;
        printf("0x%p: ", instructionAddress);
        DisassembleInstruction(hProcess, instructionAddress);
    }
}

void HandleExceptionEvent(EXCEPTION_DEBUG_INFO* pExceptionDebugInfo, HANDLE hProcess, DWORD dwThreadId, int* stepInto) {
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
    LPVOID currentInstruction = (LPVOID)(context.Rip);
#else
    LPVOID currentInstruction = (LPVOID)(context.Eip);
#endif

    printf("\nException Code: 0x%08x at address 0x%p\n", exceptionCode, currentInstruction);

    PrintInstructionsAroundIP(hProcess, currentInstruction);

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

    *stepInto = 1;

    CloseHandle(hThread);
}

void PrintRegisterState(CONTEXT* context) {
#ifdef _WIN64
    printf("RIP: 0x%llx\n", context->Rip);
    printf("RAX: 0x%llx\n", context->Rax);
    printf("RBX: 0x%llx\n", context->Rbx);
    printf("RCX: 0x%llx\n", context->Rcx);
    printf("RDX: 0x%llx\n", context->Rdx);
    printf("RSI: 0x%llx\n", context->Rsi);
    printf("RDI: 0x%llx\n", context->Rdi);
    printf("RBP: 0x%llx\n", context->Rbp);
    printf("RSP: 0x%llx\n", context->Rsp);
    printf("R8: 0x%llx\n", context->R8);
    printf("R9: 0x%llx\n", context->R9);
    printf("R10: 0x%llx\n", context->R10);
    printf("R11: 0x%llx\n", context->R11);
    printf("R12: 0x%llx\n", context->R12);
    printf("R13: 0x%llx\n", context->R13);
    printf("R14: 0x%llx\n", context->R14);
    printf("R15: 0x%llx\n", context->R15);
#else
    printf("EIP: 0x%x\n", context->Eip);
    printf("EAX: 0x%x\n", context->Eax);
    printf("EBX: 0x%x\n", context->Ebx);
    printf("ECX: 0x%x\n", context->Ecx);
    printf("EDX: 0x%x\n", context->Edx);
    printf("ESI: 0x%x\n", context->Esi);
    printf("EDI: 0x%x\n", context->Edi);
    printf("EBP: 0x%x\n", context->Ebp);
    printf("ESP: 0x%x\n", context->Esp);
#endif
}

void SingleStepDebugger(HANDLE hProcess, DWORD dwThreadId) {
    CONTEXT context;
    HANDLE hThread = OpenThread(THREAD_ALL_ACCESS, FALSE, dwThreadId);

    if (!hThread) {
        printf("Failed to open thread. Error: %lu\n", GetLastError());
        return;
    }

    context.ContextFlags = CONTEXT_CONTROL;

    while (1) {
        if (!GetThreadContext(hThread, &context)) {
            printf("Failed to get thread context. Error: %lu\n", GetLastError());
            break;
        }

#ifdef _WIN64
        context.EFlags |= 0x100;  // Set the Trap Flag (TF) for single-step
#else
        context.EFlags |= 0x100;  // Set the Trap Flag (TF) for single-step
#endif

        if (!SetThreadContext(hThread, &context)) {
            printf("Failed to set thread context. Error: %lu\n", GetLastError());
            break;
        }

        // Wait for the user to press a key to step to the next instruction
        printf("Press 's' to step into the next instruction, 'p' to print registers, or 'c' to continue: ");
        char command = getchar();
        getchar();  // Consume newline

        if (command == 'p') {
            context.ContextFlags = CONTEXT_FULL;
            if (GetThreadContext(hThread, &context)) {
                PrintRegisterState(&context);
            }
            else {
                printf("Failed to get thread context. Error: %lu\n", GetLastError());
            }
        }
        else if (command == 'c') {
            // Continue running until the next exception or breakpoint
            break;
        }
        else if (command == 'a') {
#ifdef _WIN64
            DisassembleInstruction(hProcess, (LPVOID)(context.Rip));
#else
            DisassembleInstruction(hProcess, (LPVOID)(context.Eip));
#endif
        }

        if (!ContinueDebugEvent(GetProcessId(hProcess), dwThreadId, DBG_CONTINUE)) {
            printf("Failed to continue debugging. Error: %lu\n", GetLastError());
            break;
        }
    }

    CloseHandle(hThread);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <PID>\n", argv[0]);
        return 1;
    }

    DWORD pid = atoi(argv[1]);
    DEBUG_EVENT debugEvent;
    BOOL continueDebugging = TRUE;
    DWORD continueStatus = DBG_CONTINUE;
    int stepInto = 0;

    if (!DebugActiveProcess(pid)) {
        printf("Failed to attach to process %d. Error: %lu\n", pid, GetLastError());
        return 1;
    }

    while (continueDebugging) {
        if (WaitForDebugEvent(&debugEvent, INFINITE)) {
            switch (debugEvent.dwDebugEventCode) {
            case EXCEPTION_DEBUG_EVENT:
                HandleExceptionEvent(&debugEvent.u.Exception, OpenProcess(PROCESS_ALL_ACCESS, FALSE, debugEvent.dwProcessId), debugEvent.dwThreadId, &stepInto);
                if (stepInto) {
                    SingleStepDebugger(OpenProcess(PROCESS_ALL_ACCESS, FALSE, debugEvent.dwProcessId), debugEvent.dwThreadId);
                    stepInto = 0;  // Reset step-into flag
                }
                break;

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





#if 0
#include <windows.h>
#include <stdio.h>

#define NUM_INSTRUCTIONS 5

// List of opcodes with corresponding assembly mnemonics
typedef struct {
    BYTE opcode;
    const char* mnemonic;
} OpcodeMnemonic;

OpcodeMnemonic simpleOpcodeTable[] = {
    { 0x90, "NOP" },            // No Operation
    { 0xCC, "INT 3" },          // Breakpoint Interrupt
    { 0xC3, "RET" },            // Return from Procedure
    { 0x55, "PUSH EBP" },       // Push EBP onto the stack
    { 0x89, "MOV" },            // Move Register/Memory
    { 0xE9, "JMP" },            // Jump
    { 0xEB, "JMP SHORT" },      // Short Jump
    { 0x74, "JZ" },             // Jump if Zero
    { 0x75, "JNZ" },            // Jump if Not Zero
    { 0x50, "PUSH EAX" },       // Push EAX onto the stack
    { 0x58, "POP EAX" },        // Pop EAX from the stack
    { 0xB8, "MOV EAX," },       // Move immediate to EAX
    { 0xB9, "MOV ECX," },       // Move immediate to ECX
    // Add more opcodes as needed...
};

// Helper function to read memory from the debugged process
BOOL ReadProcessMemorySafe(HANDLE hProcess, LPCVOID lpBaseAddress, LPVOID lpBuffer, SIZE_T nSize) {
    SIZE_T bytesRead;
    if (ReadProcessMemory(hProcess, lpBaseAddress, lpBuffer, nSize, &bytesRead) && bytesRead == nSize) {
        return TRUE;
    }
    return FALSE;
}

// Simple disassembler function
void DisassembleInstruction(HANDLE hProcess, LPVOID instructionAddress) {
    BYTE opcode;
    if (!ReadProcessMemorySafe(hProcess, instructionAddress, &opcode, sizeof(BYTE))) {
        printf("Failed to read instruction at 0x%p\n", instructionAddress);
        return;
    }

    // Try to find the opcode in our simple table
    const char* mnemonic = "UNKNOWN";
    for (int i = 0; i < sizeof(simpleOpcodeTable) / sizeof(OpcodeMnemonic); i++) {
        if (simpleOpcodeTable[i].opcode == opcode) {
            mnemonic = simpleOpcodeTable[i].mnemonic;
            break;
        }
    }

    printf("0x%p: %02X %s\n", instructionAddress, opcode, mnemonic);
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
        DisassembleInstruction(hProcess, instructionAddress);  // Disassemble instead of printing raw bytes
    }
}

// Function to handle exceptions and implement step-into
void HandleExceptionEvent(EXCEPTION_DEBUG_INFO* pExceptionDebugInfo, HANDLE hProcess, DWORD dwThreadId, int* stepInto) {
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

    // Enter step-into mode after an exception
    *stepInto = 1;

    CloseHandle(hThread);
}

// Function to print the current state of registers
void PrintRegisterState(CONTEXT* context) {
#ifdef _WIN64
    printf("RIP: 0x%llx\n", context->Rip);
    printf("RAX: 0x%llx\n", context->Rax);
    printf("RBX: 0x%llx\n", context->Rbx);
    printf("RCX: 0x%llx\n", context->Rcx);
    printf("RDX: 0x%llx\n", context->Rdx);
    printf("RSI: 0x%llx\n", context->Rsi);
    printf("RDI: 0x%llx\n", context->Rdi);
    printf("RBP: 0x%llx\n", context->Rbp);
    printf("RSP: 0x%llx\n", context->Rsp);
    printf("R8: 0x%llx\n", context->R8);
    printf("R9: 0x%llx\n", context->R9);
    printf("R10: 0x%llx\n", context->R10);
    printf("R11: 0x%llx\n", context->R11);
    printf("R12: 0x%llx\n", context->R12);
    printf("R13: 0x%llx\n", context->R13);
    printf("R14: 0x%llx\n", context->R14);
    printf("R15: 0x%llx\n", context->R15);
#else
    printf("EIP: 0x%x\n", context->Eip);
    printf("EAX: 0x%x\n", context->Eax);
    printf("EBX: 0x%x\n", context->Ebx);
    printf("ECX: 0x%x\n", context->Ecx);
    printf("EDX: 0x%x\n", context->Edx);
    printf("ESI: 0x%x\n", context->Esi);
    printf("EDI: 0x%x\n", context->Edi);
    printf("EBP: 0x%x\n", context->Ebp);
    printf("ESP: 0x%x\n", context->Esp);
#endif
}

// Function to interact with the debugger
void DebuggerControl(HANDLE hProcess, DWORD dwThreadId) {
    CONTEXT context;
    HANDLE hThread = OpenThread(THREAD_ALL_ACCESS, FALSE, dwThreadId);
    char command;

    if (!hThread) {
        printf("Failed to open thread. Error: %lu\n", GetLastError());
        return;
    }

    do {
        printf("Debugger> [s]tep into, [c]ontinue, [p]rint registers, [a]ssembly instruction: ");
        command = getchar();
        getchar();  // Consume newline

        if (command == 'p') {
            context.ContextFlags = CONTEXT_FULL;
            if (GetThreadContext(hThread, &context)) {
                PrintRegisterState(&context);
            }
            else {
                printf("Failed to get thread context. Error: %lu\n", GetLastError());
            }
        }
        else if (command == 's') {
            context.ContextFlags = CONTEXT_CONTROL;
            if (GetThreadContext(hThread, &context)) {
#ifdef _WIN64
                context.EFlags |= 0x100;  // Set the Trap Flag (TF) for single-step
#else
                context.EFlags |= 0x100;  // Set the Trap Flag (TF) for single-step
#endif
                if (!SetThreadContext(hThread, &context)) {
                    printf("Failed to set thread context. Error: %lu\n", GetLastError());
                }
            }
            else {
                printf("Failed to get thread context. Error: %lu\n", GetLastError());
            }
        }
        else if (command == 'a') {
            context.ContextFlags = CONTEXT_FULL;
            if (GetThreadContext(hThread, &context)) {
#ifdef _WIN64
                DisassembleInstruction(hProcess, (LPVOID)(context.Rip));
#else
                DisassembleInstruction(hProcess, (LPVOID)(context.Eip));
#endif
            }
            else {
                printf("Failed to get thread context. Error: %lu\n", GetLastError());
            }
        }

    } while (command != 'c');

    CloseHandle(hThread);
}


int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <PID>\n", argv[0]);
        return 1;
    }

    DWORD pid = atoi(argv[1]);
    DEBUG_EVENT debugEvent;
    BOOL continueDebugging = TRUE;
    DWORD continueStatus = DBG_CONTINUE;
    int stepInto = 0;

    if (!DebugActiveProcess(pid)) {
        printf("Failed to attach to process %d. Error: %lu\n", pid, GetLastError());
        return 1;
    }

    while (continueDebugging) {
        if (WaitForDebugEvent(&debugEvent, INFINITE)) {
            switch (debugEvent.dwDebugEventCode) {
            case EXCEPTION_DEBUG_EVENT:
                HandleExceptionEvent(&debugEvent.u.Exception, (HANDLE)debugEvent.dwProcessId, debugEvent.dwThreadId, &stepInto);
                if (stepInto) {
                    DebuggerControl(OpenProcess(PROCESS_ALL_ACCESS, FALSE, debugEvent.dwProcessId), debugEvent.dwThreadId);
                    stepInto = 0;  // Reset step-into flag
                }
                break;

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
void HandleExceptionEvent(EXCEPTION_DEBUG_INFO* pExceptionDebugInfo, HANDLE hProcess, DWORD dwThreadId, int* stepInto) {
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

    // Enter step-into mode after an exception
    *stepInto = 1;

    CloseHandle(hThread);
}

// Function to print the current state of registers
void PrintRegisterState(CONTEXT* context) {
#ifdef _WIN64
    printf("RIP: 0x%llx\n", context->Rip);
    printf("RAX: 0x%llx\n", context->Rax);
    printf("RBX: 0x%llx\n", context->Rbx);
    printf("RCX: 0x%llx\n", context->Rcx);
    printf("RDX: 0x%llx\n", context->Rdx);
    printf("RSI: 0x%llx\n", context->Rsi);
    printf("RDI: 0x%llx\n", context->Rdi);
    printf("RBP: 0x%llx\n", context->Rbp);
    printf("RSP: 0x%llx\n", context->Rsp);
    printf("R8: 0x%llx\n", context->R8);
    printf("R9: 0x%llx\n", context->R9);
    printf("R10: 0x%llx\n", context->R10);
    printf("R11: 0x%llx\n", context->R11);
    printf("R12: 0x%llx\n", context->R12);
    printf("R13: 0x%llx\n", context->R13);
    printf("R14: 0x%llx\n", context->R14);
    printf("R15: 0x%llx\n", context->R15);
#else
    printf("EIP: 0x%x\n", context->Eip);
    printf("EAX: 0x%x\n", context->Eax);
    printf("EBX: 0x%x\n", context->Ebx);
    printf("ECX: 0x%x\n", context->Ecx);
    printf("EDX: 0x%x\n", context->Edx);
    printf("ESI: 0x%x\n", context->Esi);
    printf("EDI: 0x%x\n", context->Edi);
    printf("EBP: 0x%x\n", context->Ebp);
    printf("ESP: 0x%x\n", context->Esp);
#endif
}

// Function to interact with the debugger
void DebuggerControl(HANDLE hProcess, DWORD dwThreadId) {
    CONTEXT context;
    HANDLE hThread = OpenThread(THREAD_ALL_ACCESS, FALSE, dwThreadId);
    char command;

    if (!hThread) {
        printf("Failed to open thread. Error: %lu\n", GetLastError());
        return;
    }

    do {
        printf("Debugger> [s]tep into, [c]ontinue, [p]rint registers: ");
        command = getchar();
        getchar();  // Consume newline

        if (command == 'p') {
            context.ContextFlags = CONTEXT_FULL;
            if (GetThreadContext(hThread, &context)) {
                PrintRegisterState(&context);
            }
            else {
                printf("Failed to get thread context. Error: %lu\n", GetLastError());
            }
        }
        else if (command == 's') {
            context.ContextFlags = CONTEXT_CONTROL;
            if (GetThreadContext(hThread, &context)) {
#ifdef _WIN64
                context.EFlags |= 0x100;  // Set the Trap Flag (TF) for single-step
#else
                context.EFlags |= 0x100;  // Set the Trap Flag (TF) for single-step
#endif
                if (!SetThreadContext(hThread, &context)) {
                    printf("Failed to set thread context. Error: %lu\n", GetLastError());
                }
            }
            else {
                printf("Failed to get thread context. Error: %lu\n", GetLastError());
            }
        }

    } while (command != 'c');

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
    int stepInto = 0;

    while (continueDebugging) {
        // Wait for a debug event
        if (WaitForDebugEvent(&debugEvent, INFINITE)) {
            DWORD continueStatus = DBG_CONTINUE;

            switch (debugEvent.dwDebugEventCode) {
            case EXCEPTION_DEBUG_EVENT:
                HandleExceptionEvent(&debugEvent.u.Exception, OpenProcess(PROCESS_VM_READ, FALSE, pid), debugEvent.dwThreadId, &stepInto);
                if (stepInto) {
                    DebuggerControl(OpenProcess(PROCESS_VM_READ, FALSE, pid), debugEvent.dwThreadId);
                }
                break;

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
void HandleExceptionEvent(EXCEPTION_DEBUG_INFO* pExceptionDebugInfo, HANDLE hProcess, DWORD dwThreadId, int* stepInto) {
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

    // Enter step-into mode after an exception
    *stepInto = 1;

    CloseHandle(hThread);
}

// Function to interact with the debugger
void DebuggerControl(HANDLE hProcess, DWORD dwThreadId) {
    CONTEXT context;
    HANDLE hThread = OpenThread(THREAD_ALL_ACCESS, FALSE, dwThreadId);
    char command;

    if (!hThread) {
        printf("Failed to open thread. Error: %lu\n", GetLastError());
        return;
    }

    do {
        printf("Debugger> [s]tep into, [c]ontinue, [p]rint context: ");
        command = getchar();
        getchar();  // Consume newline

        if (command == 'p') {
            context.ContextFlags = CONTEXT_FULL;
            if (GetThreadContext(hThread, &context)) {
#ifdef _WIN64
                printf("RIP: 0x%llx\n", context.Rip);
#else
                printf("EIP: 0x%x\n", context.Eip);
                printf("EAX 0x%x %d\n", context.Eax, context.Eax);
                printf("EBX 0x%x %d\n", context.Ebx, context.Ebx);
                printf("ECX 0x%x %d\n", context.Ecx, context.Ecx);
#endif
            }
            else {
                printf("Failed to get thread context. Error: %lu\n", GetLastError());
            }
        }
        else if (command == 's') {
            context.ContextFlags = CONTEXT_CONTROL;
            GetThreadContext(hThread, &context);
#ifdef _WIN64
            context.EFlags |= 0x100;  // Set the Trap Flag (TF) for single-step
#else
            context.EFlags |= 0x100;  // Set the Trap Flag (TF) for single-step
#endif

            //printf("0x%x %d\n", context.Eax, context.Eax);

            SetThreadContext(hThread, &context);
        }

    } while (command != 'c');

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
    int stepInto = 0;

    while (continueDebugging) {
        // Wait for a debug event
        if (WaitForDebugEvent(&debugEvent, INFINITE)) {
            DWORD continueStatus = DBG_CONTINUE;

            switch (debugEvent.dwDebugEventCode) {
            case EXCEPTION_DEBUG_EVENT:
                HandleExceptionEvent(&debugEvent.u.Exception, OpenProcess(PROCESS_VM_READ, FALSE, pid), debugEvent.dwThreadId, &stepInto);
                if (stepInto) {
                    DebuggerControl(OpenProcess(PROCESS_VM_READ, FALSE, pid), debugEvent.dwThreadId);
                }
                break;

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