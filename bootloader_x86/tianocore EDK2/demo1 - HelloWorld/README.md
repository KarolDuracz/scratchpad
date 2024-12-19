Main .c code for helloworld.efi -> https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo1%20-%20HelloWorld/HelloWorld_source_MdeModulePkg-Application-/HelloWorld/HelloWorld.c<br /><br />
.inf -> https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo1%20-%20HelloWorld/HelloWorld_source_MdeModulePkg-Application-/HelloWorld/HelloWorld.inf
<br /><br />
This is compiled code with .efi files https://github.com/KarolDuracz/scratchpad/tree/main/bootloader_x86/tianocore%20EDK2/demo1%20-%20HelloWorld/HelloWorld
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
At first I did it from the drivers windows level, I loaded sc start PrivInstDriver from demo2 about drivers. And I got a system exception. Blue screen. On the host Win 8.1 and on the virtual machine Win 10. Well, I thought, since I already have access to EDK2, why not try to display what CPUID shows on EAX, for example? Here are 3 screenshots from 3 runs, on the host, on the virtual machine and from a pendrive as raw helloworld.efi

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
