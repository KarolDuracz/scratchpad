02-11-2024  - This is not very useful, but some educational maybe for beginners like me. <br /><br />
[1]<br /><br />
<b>Setup environment</b> using vcvarsXX.bat instead create new project in MSVC. For simple case like code as below, better is to use cl.exe
to compile it. So, In my current laptop I installed MSVC 2019 in (and this is path to vcvars.bat files)
```
C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build
```
For this demo I used vcvars32.bat (When I setup x64 environment using vcvars64.bat you got error, probably 30 but I don't exatly remember ath this moment -> https://learn.microsoft.com/en-us/windows/win32/debug/system-error-codes--0-499- . But reason for what you get this error is because you ru 32 bit regiters like EAX, etc code that prepare for 32 bits environemnt. This is one of reasons. )

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple_debugger/demo2/path%20to%20vcvars%20bat.png?raw=true)

So, when we open "cmd.exe" and run this .bat script we get something like that
```
Microsoft Windows [Version 6.3.9600]
(c) 2013 Microsoft Corporation. All rights reserved.

C:\Users\kdhome>cd C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Bui
ld

C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build>vcvars32
**********************************************************************
** Visual Studio 2019 Developer Command Prompt v16.11.35
** Copyright (c) 2021 Microsoft Corporation
**********************************************************************
[vcvarsall.bat] Environment initialized for: 'x86'
```

This is important because without this setup of environent we need to link with commands like that manualy headers and libraries. This is not for this case script, but similar example how to pass through some stuff like headers etc. This is example from driver compilation "hello world".
```
@echo off
setlocal

rem Set the paths to the WDK tools
set WDK_INC=C:\Program Files (x86)\Windows Kits\8.0\Include\km
set WDK_INC2=C:\Program Files (x86)\Windows Kits\8.0\Include\km\crt
set WDK_LIB=C:\Program Files (x86)\Windows Kits\8.0\Lib\win8\km\x64

echo WDK_INC is %WDK_INC%
echo WDK_INC2 is %WDK_INC2%
echo WDK_LIB is %WDK_LIB%

rem Compile the driver
CL.exe /c /W4 /D "_AMD64_" /D "WINVER=0x0603" /D "NTDDI_VERSION=NTDDI_WINBLUE" /I "%WDK_INC%" /I "%WDK_INC2%" helloworld_kernel.c /FeHelloWorld.obj

rem Link the driver with the entry point set to DriverEntry
link.exe /driver helloworld_kernel.obj /OUT:HelloWorld.sys /LIBPATH:"%WDK_LIB%" "C:\Program Files (x86)\Windows Kits\8.0\Lib\win8\km\x64\ntoskrnl.lib" /ENTRY:DriverEntry

endlocal

```
but this is clunky

<hr>

[2]
<br /><br />
Next. <b>Setup the debugger.</b> I use WinDbg here. So. I installed Windows Kits 10.
```
C:\Program Files (x86)\Windows Kits\10\Debuggers\x86\windbg.exe 
```

For this example we will use x86 windbg. (not x64 version) - Run as administrator . But this is not necessary.

<hr >

[3]
<br /><br />
<b>Ok, time for code.</b>
<br /><br />
<i>writer.c</i> == w2.c
```
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
```
Command to compile with environment running from vcvars32.bat
```
cl.exe writer.c
```
That's it.

<br /><br /><br />

<i>reader.c</i> == r2.c
```
#include <windows.h>
#include <stdio.h>

#define SHARED_MEMORY_NAME "Local\\MySharedMemory"
#define SHARED_MEMORY_SIZE sizeof(RegisterState)

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

void read_shared_memory() {
    // Open the shared memory block created by the Writer
    HANDLE hMapFile = OpenFileMapping(FILE_MAP_READ, FALSE, SHARED_MEMORY_NAME);
    if (hMapFile == NULL) {
        printf("Could not open file mapping object (Error %d)\n", GetLastError());
        return;
    }

    // Map a view of the shared memory
    RegisterState* pBuf = (RegisterState*)MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, SHARED_MEMORY_SIZE);
    if (pBuf == NULL) {
        printf("Could not map view of file (Error %d)\n", GetLastError());
        CloseHandle(hMapFile);
        return;
    }

    printf("Reader Process: Reading register states from shared memory every second...\n");

    // Continuously read the data from shared memory every second
    while (1) {
        printf("Reader: Read register states from shared memory:\n");
        printf("EAX: %08X %d\n", pBuf->eax, pBuf->eax);
        printf("EBX: %08X\n", pBuf->ebx);
        printf("ECX: %08X\n", pBuf->ecx);
        printf("EDX: %08X\n", pBuf->edx);
        printf("ESI: %08X\n", pBuf->esi);
        printf("EDI: %08X\n", pBuf->edi);
        printf("EBP: %08X\n", pBuf->ebp);
        printf("ESP: %08X\n", pBuf->esp);
        printf("EIP: %08X %d\n", pBuf->eip, pBuf->eip);
        printf("\n");

        Sleep(100);  // Read every 1 second
    }

    // Cleanup (unreachable in this example, as it runs indefinitely)
    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);
}

int main() {
    read_shared_memory();
    return 0;
}
```
Command to compile
```
cl.exe reader.c
```

<br /><br /><br />

<i>stack.c</i> == reader.c
```
#include <windows.h>
#include <stdio.h>

#define SHARED_MEMORY_NAME "Local\\MySharedMemory"
#define SHARED_MEMORY_SIZE 512

void read_shared_memory() {
    // Open the shared memory block created by the Writer
    HANDLE hMapFile = OpenFileMapping(FILE_MAP_READ, FALSE, SHARED_MEMORY_NAME);
    if (hMapFile == NULL) {
        printf("Could not open file mapping object (Error %d)\n", GetLastError());
        return;
    }

    // Map a view of the shared memory
    char* pBuf = (char*)MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, SHARED_MEMORY_SIZE);
    if (pBuf == NULL) {
        printf("Could not map view of file (Error %d)\n", GetLastError());
        CloseHandle(hMapFile);
        return;
    }

    printf("Reader Process: Reading data from shared memory every second...\n");

    // Continuously read the data from shared memory every second
    while (1) {
        printf("Reader: Read from shared memory: ");
        for (int i = 0; i < SHARED_MEMORY_SIZE; i++) {
            printf("|%02X %c", (unsigned char)pBuf[i], (unsigned char)pBuf[i]);  // Print as hex
        }
        printf("\n");
        Sleep(1000);  // Read every 1 second
    }

    // Cleanup (unreachable in this example, as it runs indefinitely)
    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);
}

int main() {
    read_shared_memory();
    return 0;
}
```
Command to compile
```
cl.exe stack.c
```

<hr>

<h1>FIRST RUN</h1>
As you see in write.c code in line from 43 you see that

```
    // Continuously get the register states
    while (1) {
		
		Sleep(10 * 1000);
		__asm {
			int 3
		}
```

That means we have 10 seconds (we can change it) to start the debugger. This is important. So:<br />
<b>1. Start WinDbg x86 version as administrator. And run write.exe file.</b><br />
<b>2. You have 10 seconds to find (using F6 key) process "write.exe" and to attach debugger to this process</b><br />
<b>3. When you attached debugger and it running you see something like this</b><br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple_debugger/demo2/in%20my%20example%20write%20equals%20w2%20exe.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple_debugger/demo2/after%20attached.png?raw=true)


<b>4. Run reader.exe</b><br />
<b>5. Run stack.exe</b><br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple_debugger/demo2/step%205.png?raw=true)

<b>6. In windbg hit 3 times "g" command to run debugger</b><br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple_debugger/demo2/3x%20g%20command.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple_debugger/demo2/output.png?raw=true)

https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/controlling-the-target<br />
https://learn.microsoft.com/en-us/windows-hardware/drivers/debuggercmds/p--step-<br />
https://codemachine.com/articles/windbg_quickstart.html <--- quick tutorial how to use WINDBG<br />

<h1>What is that? What this demonstrate?</h1>
The main goal it to attach any process, for example start from calc.exe and notepad.exe and  have more control over the context and execution of code. But in here I start from simple assembler code inside debugger to explore and learn bahviours like this and more.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple_debugger/demo2/the%20end.png?raw=true)

<hr>
<br /><br />
This is example code how to write data to shared memory for this stack.c . This was basic example. But here we don't put registers like on the images and code above above, but we get heap from guest process. In here i write into "MySharedMemory" but replace to previus and compile it again. Change SHARED_MEMORY_SIZE to 512. 

```
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
```

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple_debugger/demo2/last%20example.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple_debugger/demo2/stack%20print%20vs%20windbg.png?raw=true)

<hr>
On the Windows 7 installed on Virtualbox also run properly. But you need compile it again as x86 exe. Setup x86 environment for cl.exe and compile without "int 3" instructions inside code! Because In this case I don't installed WinDbg on this Win7. Comile r2.c and w2.c. This demo shown live state of registers. And we have data inside share memory about it after 100 ms in this case.
<br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple_debugger/demo2/68%20-%204-11-2024%20-%20demo%20win7%20virtual%20box.png?raw=true)
