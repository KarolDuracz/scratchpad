<h2>Demo2</h2>
Execute unprivileged instruction to read MSR and CR0 state. 
<br /><br />


![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Hello%20World%20Drivers/demo2/17112024%20-%20pic1%20-%20read%20msr%20and%20cr0.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Hello%20World%20Drivers/demo2/17112024%20-%20pic2%20-%20EFER%20status.png?raw=true)

<h2>How to run</h2>
1. Follow this instruction how to reboot (https://github.com/KarolDuracz/scratchpad/tree/main/Hello%20World%20Drivers/Windows/demo1) - disable-enable-driver-signature-enforcement-on-windows<br />
2. Execute "run.bat" script. This compile all things (only without readmsr.asm - this is not needed here) - I use here __readcr0 (https://learn.microsoft.com/en-us/cpp/intrinsics/readcr0?view=msvc-170) and  __readmsr (https://learn.microsoft.com/en-us/cpp/intrinsics/readmsr?view=msvc-170) <br />
3. Create service (cmd as administrator)

```
sc create PrivInstDriver binPath= "C:\Users\kdhome\Documents\progs\__trash-22-10-2024-startfrom\17-11-2024\PrivilegedInstructionsDriver.sys" type= kernel
```
4. Open Dbgview (not dbgview64) - SysinternalsSuite tools (https://learn.microsoft.com/en-us/sysinternals/downloads/sysinternals-suite)
5. start service PrivInstDriver

```
sc start PrivInstDriver
```

6. sc stop PrivInstDriver - to stop
7. sc delete PrivInstDriver - to delete
8. If you want to run again -> open run.bat > then create service "sc create..." > then watch DbgView > start service ... 

<hr>
In "privileged.asm" you can execute a sequence of asm code and do an "insert" into the main code as shown by the CLI operation. You just have to do RET because there is a CALL before. But on this picture I do test to check EFLAGS (https://en.wikipedia.org/wiki/FLAGS_register) using (https://learn.microsoft.com/en-us/cpp/intrinsics/readeflags?view=msvc-170). This is only introduce... how to get access to these instructions. But EFLAGS in this demo don't show IOPL bit. Only first 10 bit max, for first test 0x282 and for second after CLI 0x82. This is only bytes 0-9.<br /><br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Hello%20World%20Drivers/demo2/17112024%20-%20pic3%20-%20eflags.png?raw=true)

<hr>
<h2>How to pass extern variable form .asm to .c program</h2>

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Hello%20World%20Drivers/demo2/17112024%20-%20pic4%20-%20extern%20variable%20pass%20to%20main%20c%20program.png?raw=true)

run.bat - only line to MSVC=C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\14.29.30133\lib\x64 to fix LINK : fatal error LNK1104: cannot open file 'LIBCMT.lib'

```
@echo off
setlocal

rem Set the paths to the WDK tools
set WDK_INC=C:\Program Files (x86)\Windows Kits\8.0\Include\km
set WDK_INC_CRT=C:\Program Files (x86)\Windows Kits\8.0\Include\km\crt
set WDK_LIB=C:\Program Files (x86)\Windows Kits\8.0\Lib\win8\km\x64
set WDK_LIB_CRT=C:\Program Files (x86)\Windows Kits\8.0\Lib\win8\um\x64
set WDK_BIN=C:\Program Files (x86)\Windows Kits\8.0\bin\x64
set MSVC=C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\14.29.30133\lib\x64

echo WDK_INC is %WDK_INC%
echo WDK_LIB is %WDK_LIB%
echo WDK_LIB_CRT is %WDK_LIB_CRT%
echo MSVC is %MSVC%

rem Compile the driver
CL.exe /c /W4 /D "_AMD64_" /D "WINVER=0x0603" /D "NTDDI_VERSION=NTDDI_WINBLUE" /I "%WDK_INC%" /I "%WDK_INC_CRT%" PrivilegedInstructionsDriver.c /FePrivilegedInstructionsDriver.obj

rem Assemble the privileged instructions file
ml64.exe /c privileged.asm /Fo:privileged.obj

rem Link the driver
link.exe /driver PrivilegedInstructionsDriver.obj privileged.obj /OUT:PrivilegedInstructionsDriver.sys /LIBPATH:"%WDK_LIB%" /LIBPATH:"%WDK_LIB_CRT%" /LIBPATH:"%MSVC%" ntoskrnl.lib /ENTRY:DriverEntry /SUBSYSTEM:NATIVE

endlocal
```

PrivilegedInstructionsDriver.c

```
#include <ntddk.h> // WDK headers

// Function prototypes
VOID UnloadRoutine(_In_ PDRIVER_OBJECT DriverObject);
VOID ExecutePrivilegedInstructions(void);

// Declare external assembly function
extern void DoCli(void);

// extern 
extern unsigned __int64 _vartest;

// DriverEntry function
NTSTATUS DriverEntry(
    _In_ PDRIVER_OBJECT DriverObject,
    _In_ PUNICODE_STRING RegistryPath)
{
    UNREFERENCED_PARAMETER(RegistryPath);

    KdPrint(("PrivilegedInstructionsDriver loaded.\n"));

    // Set unload routine
    DriverObject->DriverUnload = UnloadRoutine;

    // Execute privileged instructions
    ExecutePrivilegedInstructions();

    return STATUS_SUCCESS;
}

// Unload routine
VOID UnloadRoutine(_In_ PDRIVER_OBJECT DriverObject)
{
    UNREFERENCED_PARAMETER(DriverObject);
    DbgPrint(("PrivilegedInstructionsDriver unloaded.\n"));
}

// Function to execute privileged instructions
VOID ExecutePrivilegedInstructions(void)
{
    unsigned __int64 msrValue;
	unsigned __int64 cr0Value;
	unsigned __int64 eflgas_state;
	unsigned __int64 cr4;

    // Example: Read from MSR 0xC0000080 (EFER - Extended Feature Enable Register)
    msrValue = __readmsr(0xC0000080); // MSR intrinsic
	DbgPrint("READ test");
    //DbgPrint(("EFER MSR Value: 0x%llx\n", msrValue));
	//DbgPrint(msrValue);
	DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "EFER MSR Value: 0x%llx\n", msrValue);
	DbgPrint("READ test end");
	
	DbgPrint("READ CR0");
	cr0Value = __readcr0(); 
	DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "CR 0 status Value: 0x%llx\n", cr0Value);
	
	DbgPrint("READ CR4");
	cr4 = __readcr4(); 
	DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "CR 4 status Value: 0x%llx\n", cr4);
	
	DbgPrint("eflags");
	eflgas_state = __readeflags();
	DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "eflags status Value: 0x%llx\n", eflgas_state);
	
	

    // Example: Attempt CLI (privileged instruction)
    __try {
        DoCli(); // Call external assembly function
		
		DbgPrint("eflags");
		eflgas_state = __readeflags();
		DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "eflags status Value: 0x%llx\n", eflgas_state);
		
		DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "_vartest: 0x%llx\n", _vartest);
		
        DbgPrint(("CLI instruction executed successfully.\n"));
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        DbgPrint(("CLI instruction caused a privileged instruction exception.\n"));
    }

    // Example: Write to MSR (if required)
    __writemsr(0xC0000080, msrValue); // Write the same value back
    DbgPrint(("EFER MSR updated (write test).\n"));
	

	
}
```

privileged.asm

```
.DATA
    PUBLIC _vartest         ; Make _vartest a public symbol
    _vartest QWORD 1234h    ; Define _vartest as a 64-bit variable initialized to 0x1234

.CODE

    PUBLIC DoCli            ; Export the DoCli function
DoCli PROC
    cli                     ; Clear the interrupt flag
    PUSHFQ                  ; Push the flags register onto the stack
    POP RAX                 ; Pop the flags into RAX
    MOV [_vartest], RAX       ; Store the value of RAX in _vartest
    RET
DoCli ENDP

END
```

btw. When you put ``` DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "_vartest: 0x%llx\n", _vartest); ``` 
before DoCli, you will see 0x1234 value. So this is important for test if this works and get correct value into this _vartest from the stack.
<br />

<hr>
https://www.singlix.com/trdos/archive/OSDev_Wiki/IOPL.pdf - push, pop eflags example <br />
link to intel manual - Intel® 64 and IA-32 Architectures
Software Developer’s Manual
Combined Volumes:
1, 2A, 2B, 2C, 2D, 3A, 3B, 3C, 3D, and 4<br />
https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjYof-DlOOJAxVQIRAIHdhkG9oQFnoECBQQAQ&url=https%3A%2F%2Fcdrdv2-public.intel.com%2F671200%2F325462-sdm-vol-1-2abcd-3abcd.pdf&usg=AOvVaw004y3Ieq_CROz-BRxT08g1&opi=89978449
