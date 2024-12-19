Main .c code for helloworld.efi -> https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo1%20-%20HelloWorld/HelloWorld_source_MdeModulePkg-Application-/HelloWorld/HelloWorld.c<br /><br />
.inf -> https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo1%20-%20HelloWorld/HelloWorld_source_MdeModulePkg-Application-/HelloWorld/HelloWorld.inf
<br /><br />
This is compiled code with .efi files https://github.com/KarolDuracz/scratchpad/tree/main/bootloader_x86/tianocore%20EDK2/demo1%20-%20HelloWorld/HelloWorld <br />
<h2>Compilation</h2>
Compiling - This compiles just this helloworld example and just produces a .efi for it. Nothing else. This is to speed up the compilation process a bit and not compile the entire MdeModulePkg module.<br /><br />

```
cd edk2 // main edk2 folder, go here first
edksetup.bat // setup environemnt - read guide for edk2 installation and configuration which I wrote here
build -p MdeModulePkg/MdeModulePkg.dsc -m MdeModulePkg/Application/HelloWorld/HelloWorld.inf // compile via .inf just like that
```

<h2>Explanation</h2>
I thought I'd throw here HelloWorld.c code for EFI also. Because I started getting deeper into the CPU configuration, and I can't read certain MSR regions. Here is an example of how to read some registers from MSR. I can read 0xC0000080 https://github.com/KarolDuracz/scratchpad/tree/main/Hello%20World%20Drivers/demo2 which corespond do AMD but Intel support this, but I can' read A32_MPERF (Addr: E7H) and IA32_APERF (E8H) etc like IA32_PERF_CTL . This is confusing ... 
<br /><br />
Few links: <br />
[1] https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-vol-3b-part-2-manual.pdf <br />
[2] https://github.com/biosbits/bits/tree/master <br />
[3] https://community.intel.com/t5/Software-Tuning-Performance/How-to-determine-cause-of-processor-frequency-scale-down-to-200/m-p/1137067 <br />
<br /><br />
This is confusing, because according to what CPUID shows I have these bits set which allow reading MSR or features via CPUID(0x6). Maybe I'm doing something wrong or I don't know something yet...
<br /><br />
At first I did it from the drivers windows level, I loaded sc start PrivInstDriver from demo2 about drivers. And I got a system exception. Blue screen. On the host Win 8.1 and on the virtual machine Win 10. Well, I thought, since I already have access to EDK2, why not try to display what CPUID shows on EAX, for example? Here are 3 screenshots from 3 runs, on the host, on the virtual machine and from a pendrive as raw helloworld.efi<br /><br />

HOST WIN 8.1 - 0x75

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo1%20-%20HelloWorld/212%20-%2019-12-2024%20-%20win%2081%20host%200x75.png?raw=true)

VIRTUAL MACHINE WIN 10 - 0x4

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo1%20-%20HelloWorld/211%20-%2019-12-2024%20-%20test%20win10%20virtual%20machine.png?raw=true)

PENDRIVE BOOT WITHOUT OS - helloworld.efi from BIOS - 0x75

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo1%20-%20HelloWorld/1734647249018.jpg?raw=true)

CPUID EAX(0x6) - On EFI I checked EAX, here EDX bit 0, but for comparison this is also a screenshot of what the user-space application shows when compiling and running this code on Win 8.1 host machine
```
#include <Windows.h>
#include <stdio.h>
#include <intrin.h> // For __cpuid intrinsic

int main()
{

	int cpuInfo[4];
	__cpuid(cpuInfo, 0x06);

	printf("%d %d \n", (cpuInfo[2] & (1 << 0)), cpuInfo[2]);


	return 0;
}
```

Output - 0b1001

```
1 9
```

<h2>Summary</h2>
This is introduce and example of hello world to read certain registers from this manual https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-vol-3b-part-2-manual.pdf

<h2>Some examples how to read something</h2>

How to use AsmReadMsr64 maybe 
```
#include <PiPei.h>
#include <Library/DebugLib.h>
#include <Library/MsrLib.h>

EFI_STATUS
EFIAPI
MyPeimEntryPoint (
  IN EFI_PEI_FILE_HANDLE FileHandle,
  IN CONST EFI_PEI_SERVICES **PeiServices
  )
{
    UINT64 mperf, aperf;

    // Read IA32_MPERF and IA32_APERF MSRs
    mperf = AsmReadMsr64(0xE7);
    aperf = AsmReadMsr64(0xE8);

    // Output to debug log
    DEBUG((DEBUG_INFO, "IA32_MPERF: 0x%lx\n", mperf));
    DEBUG((DEBUG_INFO, "IA32_APERF: 0x%lx\n", aperf));

    return EFI_SUCCESS;
}

```

How to read CPUID, but for me I can't configure library for DEBUG that's why I used this style from third code on the bottom with buffer and *hexchar
```
#include <Library/BaseLib.h>
#include <Library/DebugLib.h>

VOID CheckPStateSupport(VOID)
{
    UINT32 Eax, Ebx, Ecx, Edx;

    // Execute CPUID with EAX=0x06
    AsmCpuid(0x06, &Eax, &Ebx, &Ecx, &Edx);

    if (Ecx & 0x1) {
        DEBUG((DEBUG_INFO, "P-State feedback supported (IA32_MPERF & IA32_APERF).\n"));
    } else {
        DEBUG((DEBUG_INFO, "P-State feedback NOT supported.\n"));
    }
}
```

third which I used here - read CPUID from efi level

```
UINT32 eax, ebx, ecx, edx;
  AsmCpuid(0x06, &eax, &ebx, &ecx, &edx);
  
  
  
  CHAR16 buffer[64];
  //UnicodeSPrint(buffer, sizeof(buffer), L"eax: 0x%08x\n", eax);
  //UnicodeValueToString(buffer, LEFT_JUSTIFY, eax, 16);
  
  CHAR16 *hexchar = L"01234567890ABCDEF";
  
  for (INTN i = 0; i < 8; i++) {
	buffer[7 - i] = hexchar[(eax >> (i * 4)) & 0xf];
  }
  buffer[8] = L'\0';
  
  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer);
```

<br /><br />
<h2>Back to driver level</h2>
https://github.com/KarolDuracz/scratchpad/tree/main/Hello%20World%20Drivers
This msrValue = __readmsr(0xe7);  produce exception and blue screen for my case.

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
	
	// 19-12-2024 Read some power management 
	msrValue = __readmsr(0xe7); 
	DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "Value: 0x%llx\n", msrValue);

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
