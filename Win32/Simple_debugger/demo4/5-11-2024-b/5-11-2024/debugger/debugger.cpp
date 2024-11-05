#include <windows.h>
#include <iostream>
#include <stdio.h>

DWORD_PTR dispatchMessageAddr = 0x00007ffd2a8823a0; // Replace with the address of user32!DispatchMessageW in the target process

// Function to set a breakpoint by modifying the memory at the given address
bool SetBreakpoint(HANDLE hProcess, DWORD_PTR address, BYTE &originalByte) {
    SIZE_T bytesRead;
    if (!ReadProcessMemory(hProcess, (LPCVOID)address, &originalByte, 1, &bytesRead) || bytesRead != 1) {
        return false;
    }

    // Write INT3 (0xCC) to trigger breakpoint
    BYTE int3 = 0xCC;
    SIZE_T bytesWritten;
    return WriteProcessMemory(hProcess, (LPVOID)address, &int3, 1, &bytesWritten) && bytesWritten == 1;
}

// Restore the original byte to remove the breakpoint
bool RemoveBreakpoint(HANDLE hProcess, DWORD_PTR address, BYTE originalByte) {
    SIZE_T bytesWritten;
    return WriteProcessMemory(hProcess, (LPVOID)address, &originalByte, 1, &bytesWritten) && bytesWritten == 1;
}

// Debugger main loop
void DebugLoop(DWORD processId) {
    if (!DebugActiveProcess(processId)) {
        std::cerr << "Failed to attach to process." << std::endl;
        return;
    }

    HANDLE hProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, processId);
    if (!hProcess) {
        std::cerr << "Failed to open process." << std::endl;
        return;
    }

    BYTE originalByte;
    bool breakpointSet = SetBreakpoint(hProcess, dispatchMessageAddr, originalByte);
    if (!breakpointSet) {
        std::cerr << "Failed to set breakpoint on DispatchMessageW." << std::endl;
        return;
    }

    DEBUG_EVENT debugEvent;
    CONTEXT context = {0};
    context.ContextFlags = CONTEXT_FULL;

    while (true) {
        if (!WaitForDebugEvent(&debugEvent, INFINITE))
            break;

        DWORD continueStatus = DBG_CONTINUE;

        if (debugEvent.dwDebugEventCode == EXCEPTION_DEBUG_EVENT) {
            EXCEPTION_DEBUG_INFO &exception = debugEvent.u.Exception;
            if (exception.ExceptionRecord.ExceptionCode == EXCEPTION_BREAKPOINT) {
                
                // Check if this is the DispatchMessageW breakpoint
                if ((DWORD_PTR)exception.ExceptionRecord.ExceptionAddress == dispatchMessageAddr) {
                    std::cout << "Breakpoint hit at DispatchMessageW." << std::endl;

                    // Remove DispatchMessageW breakpoint temporarily
                    RemoveBreakpoint(hProcess, dispatchMessageAddr, originalByte);
					
					std::cout << "test " << std::endl;

                    // Get the thread context to read RCX
                    HANDLE hThread = OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT, FALSE, debugEvent.dwThreadId);
                    if (hThread) {
						
						std::cout << "enter to thread context [1] | thread id from event comming : " << debugEvent.dwThreadId << std::endl;
						
                        if (GetThreadContext(hThread, &context)) {
							
							std::cout << "enter to thread context [2] " << std::endl;
							
							std::cout << " ========== REGISTER ============== " << std::endl;
							std::cout << " rax " <<  std::hex << context.Rax << std::endl;
							std::cout << " rbx "  << std::hex << context.Rbx << std::endl;
							std::cout << " rcx "  << std::hex << context.Rcx << std::endl;
							std::cout << " rdx "  << std::hex << context.Rdx << std::endl;
							std::cout << " rsi " << std::hex << context.Rsi << std::endl;
							std::cout << " rdi "  << std::hex << context.Rdi << std::endl;
							std::cout << " rsp "   << std::hex << context.Rsp << std::endl;
							std::cout << " rbp "<< std::hex << context.Rbp << std::endl;
							std::cout << " RIP "<< std::hex << context.Rip << std::endl;
							
                            DWORD_PTR rcxValue = context.Rcx + 0x8;
							
							std::cout << rcxValue << std::endl;
							std::cout << std::hex << rcxValue << std::endl;
							
							// Read from rcxValue + offset
							printf("-----\n");
							char ret[64];
                            SIZE_T ret_bytes;
                            if (ReadProcessMemory(hProcess, (LPCVOID)rcxValue, &ret, 64, &ret_bytes)) {								
                               for (int i = 0; i < 64; i++) {									
									printf("%d %x |\t %c \t %d \t 0x%.2X  \t\t\t %d  \n", i, i, ret[i], ret[i], ret[i], (ret[i]==0));	
							   }
                            }
							printf("-----\n");
							
							// print 
							/*
							HWND   hwnd;      // Offset +0x00
							UINT   message;   // Offset +0x08 (64-bit) / +0x04 (32-bit)
							WPARAM wParam;    // Offset +0x10 (64-bit) / +0x08 (32-bit)
							LPARAM lParam;    // Offset +0x18 (64-bit) / +0x0C (32-bit)
							DWORD  time;      // Offset +0x20 (64-bit) / +0x10 (32-bit)
							POINT  pt;        // Offset +0x24 (X) and +0x28 (Y) on 64-bit, +0x14 and +0x18 on 32-bit
							
							typedef struct tagMSG {
								HWND   hwnd;      // Window handle
								UINT   message;   // Message ID
								WPARAM wParam;    // Additional message information
								LPARAM lParam;    // Additional message information
								DWORD  time;      // Timestamp
								POINT  pt;        // Cursor position
							} MSG, *PMSG;
							
							user32!tagMSG
						   +0x000 hwnd            : 0x12345678 HWND
						   +0x008 message         : 0x201 (WM_LBUTTONDOWN)
						   +0x010 wParam          : 0x1
						   +0x018 lParam          : 0x20001
						   +0x020 time            : 123456789
						   +0x024 pt              : _POINT ( X = 0x100, Y = 0x200 )
							*/

							std::cout << " ======================== " << std::endl;
							std::cout << std::hex << context.Rcx + 0x0 << std::endl;
							std::cout << std::hex << context.Rcx + 0x8 << std::endl;
							std::cout << std::hex << context.Rcx + 0x10 << std::endl;
							std::cout << std::hex << context.Rcx + 0x18 << std::endl;
							std::cout << std::hex << context.Rcx + 0x20 << std::endl;
							std::cout << std::hex << context.Rcx + 0x24 << std::endl;
							

                            // Read the value at RCX + 0x8
                            DWORD_PTR valueAtOffset;
                            SIZE_T bytesRead;
                            if (ReadProcessMemory(hProcess, (LPCVOID)rcxValue, &valueAtOffset, sizeof(valueAtOffset), &bytesRead) && bytesRead == sizeof(valueAtOffset)) {
                                if (valueAtOffset == 201) {
                                    std::cout << "Condition met: *(RCX + 0x8) == 201" << std::endl;
                                }
                            }
							
							std::cout << " value offset : " << valueAtOffset << std::endl;
							
                        }
                        CloseHandle(hThread);
                    }

                    // Re-set DispatchMessageW breakpoint
                    SetBreakpoint(hProcess, dispatchMessageAddr, originalByte);
                }
            }
        }

        ContinueDebugEvent(debugEvent.dwProcessId, debugEvent.dwThreadId, continueStatus);
    }

    DebugActiveProcessStop(processId);
    CloseHandle(hProcess);
}

int main() {
    DWORD processId;
    std::cout << "Enter target process ID: ";
    std::cin >> processId;

    DebugLoop(processId);

    return 0;
}
