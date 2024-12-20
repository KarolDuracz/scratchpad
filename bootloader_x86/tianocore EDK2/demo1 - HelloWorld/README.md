<h2>TO FIX</h2>
Go to bottom of this page. the code to compile works, but there is a bad implementation of the decimal to string as hex converter. In this form it can be compiled and run, but it may display wrong values. <b>OK, a few things still need to be checked so that I can display meaningful information. I did it quickly. But what is important here is that there is a folder with the source code. And a description of how I compiled and ran it. This is important here in this demo1. The rest of the details can be fixed.</b> <br />I just changed what is default here ->
https://github.com/tianocore/edk2/tree/master/MdeModulePkg/Application/HelloWorld
<br />
<h2>Code source</h2>
Main .c code for helloworld.efi -> https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo1%20-%20HelloWorld/HelloWorld_source_MdeModulePkg-Application-/HelloWorld/HelloWorld.c<br /><br />
.inf -> https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo1%20-%20HelloWorld/HelloWorld_source_MdeModulePkg-Application-/HelloWorld/HelloWorld.inf
<br /><br />
This is compiled code with .efi files https://github.com/KarolDuracz/scratchpad/tree/main/bootloader_x86/tianocore%20EDK2/demo1%20-%20HelloWorld/HelloWorld <br />
<h2>Compilation</h2>
Compiling - This compiles just this helloworld example and just produces a .efi for it. Nothing else. This is to speed up the compilation process a bit and not compile the entire MdeModulePkg module.<br /><br />

```
// main edk2 folder, go here first
cd edk2

// setup environemnt - read guide for edk2 installation and configuration which I wrote here
edksetup.bat

// compile via .inf just like that
build -p MdeModulePkg/MdeModulePkg.dsc -m MdeModulePkg/Application/HelloWorld/HelloWorld.inf 
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

<h2>Disassembly EFI - how looks like assembler</h2>

Most efi or modules downloaded from a tool like EFI Tool It has a subsystem set up to 0xb (11). So to run in the emulator you need to change this bit in the process image, e.g. use python to open the file open(file, "rb").read() and save it to a temporary array as to_bytes()... this is how looks like pseudo code:

```
f = open("path to .efi", "rb").read()
arr=[]
idx=0
for i in f:
  if i == 0xb:
    Print(idx) // this is to find all elements compare to 0xb
  arr.append(i) // add elements to array 
  idx += 1
// change particular index in process image for example byte 267 0xb > 0xa
arr[267] = 0xa
// write to file from arr
f2 = open("path to new .efi", "wb")
for k in arr:
  // convert class int to bytes
  f2.write(k.to_bytes(1, 'big')))
f2.close()
```


![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo1%20-%20HelloWorld/disasm%20ida%201.png?raw=true)


![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo1%20-%20HelloWorld/disasm%20ida%202.png?raw=true)


![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo1%20-%20HelloWorld/195%20-%2018-12-2024%20-%20no%20to%20juz%20chyba%20mam%20dlaczego%20nie%20dziala.png?raw=true)

<hr>
<h2>to fix...</h2>
To fix...part which I used in helloworld.efi to print registers values probably it probably has a bad implementation. But I'm not 100% sure right now. I'm leaving this as information here. That means something is wrong. But that's not important now. I'm just leaving it to double check converting DECIMAL to HEX AS STRING. It's trivial. I'm just posting it as a TODO.


```
int eax = 3735929054; //  0xdeadc0de;
	const wchar_t* hexchar = L"01234567890ABCDEF";
	wchar_t buffer[64];

	for (int i = 0; i < 8; i++) {
		printf("%c %x  %x \n", hexchar[(eax >> (i * 4)) & 0xf], ((eax >> (i * 4))), ((eax >> (i * 4)) & 0xf));
		buffer[7 - i] = hexchar[(eax >> (i * 4)) & 0xf];
	}
	buffer[8] = L'\0';
```

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo1%20-%20HelloWorld/to%20fix.png?raw=true)

```
#include <Windows.h>
#include <stdio.h>
#include <intrin.h> // For __cpuid intrinsic

int main()
{

	int cpuInfo[4];
	__cpuid(cpuInfo, 0x06);

	printf("%d %d \n", (cpuInfo[2] & (1 << 0)), cpuInfo[2]);

	int retVal;
	const char* buf = "0xdeadc0de";
	retVal = (int)strtol(buf, NULL, 16);
	printf("Converted value: %d (0x%x)\n", retVal, retVal);

	const wchar_t* hexchar1 = L"0123456789ABCDEF";
	wchar_t buffer1[] = L"DEADC0DE";
	buffer1[8] = L'\0'; // Ensure null termination
	int eax1 = 0; // Resulting integer
	for (int i = 0; i < 8; i++) {
		// Find the numeric value of the character in the hexchar array
		wchar_t c = buffer1[i];
		int value = 0;
		for (int j = 0; j < 16; j++) {
			if (hexchar1[j] == c) {
				value = j;
				break;
			}
		}

		// Accumulate the result
		eax1 = (eax1 << 4) | value;
		printf("Character: %lc, Value: %x, EAX: %x\n", c, value, eax1);
	}

	printf("Final result: %d (0x%x)\n", eax1, eax1);

	int eax = 3735929054; //  0xdeadc0de;
	const wchar_t* hexchar = L"01234567890ABCDEF";
	wchar_t buffer[64];

	for (int i = 0; i < 8; i++) {
		printf("%c %x  %x \n", hexchar[(eax >> (i * 4)) & 0xf], ((eax >> (i * 4))), ((eax >> (i * 4)) & 0xf));
		buffer[7 - i] = hexchar[(eax >> (i * 4)) & 0xf];
	}
	buffer[8] = L'\0';

	printf("%ws \n", buffer);

	return 0;
}
```

The problem is that I'm just starting with EDK2. I don't really know what types are and how to convert them correctly. There are a few methods to convert integer (register value) > hex as string, int to bin as string etc. This is another example of how to convert to binary instead of hex, but it still needs to be as a string if you want to use in SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer); as parameter . This is only another example HOW TO FIX THAT... <br />
https://learn.microsoft.com/pl-pl/cpp/cpp/char-wchar-t-char16-t-char32-t?view=msvc-170

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo1%20-%20HelloWorld/__to_fix_some_example_how_to_do.png?raw=true)

```
#include <Windows.h>
#include <stdio.h>
#include <intrin.h> // For __cpuid intrinsic
#include <stdlib.h>


VOID PrintAsBinary(UINT32 value, __int16* buffer) {
	buffer[0] = L'0';
	buffer[1] = L'b'; // Binary prefix

	for (UINT32 i = 0; i < 32; i++) {
		// Extract each bit and store it in the buffer
		buffer[2 + i] = (value & (1U << (31 - i))) ? L'1' : L'0';
	}

	buffer[34] = L'\0'; // Null-terminate the string
}


VOID PrintAsBinary_standard_types(int value, char* buffer) {
	buffer[0] = L'0';
	buffer[1] = L'b'; // Binary prefix

	for (int i = 0; i < 32; i++) {
		// Extract each bit and store it in the buffer
		buffer[2 + i] = (value & (1U << (31 - i))) ? L'1' : L'0';
	}

	buffer[34] = L'\0'; // Null-terminate the string
}

/*
 * Original implementation in helloworld.efi and types 
 *
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
*/

int main()
{

	int cpuInfo[4];
	__cpuid(cpuInfo, 0x06);

	printf("%d %d \n", (cpuInfo[2] & (1 << 0)), cpuInfo[2]);

	int retVal;
	const char* buf = "0xdeadc0de";
	retVal = (int)strtol(buf, NULL, 16);
	printf("Converted value: %d (0x%x)\n", retVal, retVal);

	const wchar_t* hexchar1 = L"0123456789ABCDEF";
	wchar_t buffer1[] = L"DEADC0DE";
	buffer1[8] = L'\0'; // Ensure null termination
	int eax1 = 0; // Resulting integer
	for (int i = 0; i < 8; i++) {
		// Find the numeric value of the character in the hexchar array
		wchar_t c = buffer1[i];
		int value = 0;
		for (int j = 0; j < 16; j++) {
			if (hexchar1[j] == c) {
				value = j;
				break;
			}
		}

		// Accumulate the result
		eax1 = (eax1 << 4) | value;
		printf("Character: %lc, Value: %x, EAX: %x\n", c, value, eax1);
	}

	printf("Final result: %d (0x%x)\n", eax1, eax1);

	int eax = 3735929054; //  0xdeadc0de;
	const wchar_t* hexchar = L"01234567890ABCDEF";
	wchar_t buffer[64];

	for (int i = 0; i < 8; i++) {
		printf("%c %x  %x \n", hexchar[(eax >> (i * 4)) & 0xf], ((eax >> (i * 4))), ((eax >> (i * 4)) & 0xf));
		buffer[7 - i] = hexchar[(eax >> (i * 4)) & 0xf];
	}
	buffer[8] = L'\0';

	printf("%ws \n", buffer);

	//
	for (int i = 0; i < 50; i++) {
		printf("-");
	}
	printf("\n");
	//

	int val = 0xff;
	__int16 buf1[64];

	PrintAsBinary(val, buf1);

	printf("1: %s \n", buf1);
	printf("2: %ws \n", buf1);

	int val1 = 0xff;
	char buf2[64];

	PrintAsBinary_standard_types(val1, buf2);

	printf("3: %s \n", buf2);
	printf("4: %ws \n", buf2);

	int dead_val = 0xdeadc0de;

	PrintAsBinary_standard_types(dead_val, buf2);

	printf("5: %s \n", buf2);
	printf("6: %ws \n", buf2);

	return 0;
}
```

I added it here to reference to the hello world I wrote about when describing how I installed EDK2. I'm not messing with it for now, so I'm leaving it as it is, but I know someone is watching and reading it, so I'm posting what's there...
<br /><br />
That's all for demo1.
<hr>
<br /><br />
But at a certain stage the question arises - is what someone has implemented really a good solution to the essence of the problem? just think about all this for a moment instead of copying and pasting. Times are changing, today we need different tools than 100, 80, 60, 40, 20 years ago... 
<hr>
<h2>other projects based on EDK2</h2>
[1] This looks interesting :  https://github.com/slimbootloader/slimbootloader/tree/master   ---> docs https://slimbootloader.github.io/ - Nice and clear explanation and code. Configuration examples for some platforms.
