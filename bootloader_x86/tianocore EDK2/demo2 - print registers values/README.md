<h2>Demo2 - check and fix errors from demo1 and move forward some things</h2>

[ 1 ] - <b>First, I checked these errors from demo 1</b> - https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo1%20-%20HelloWorld/HelloWorld_source_MdeModulePkg-Application-/HelloWorld/HelloWorld.c#L63 <br />
And it works, but in line 63 is two zeros. 0123456789(0)ABCDEF . Just remove the zero from the middle part. 0123456789ABCDEF.
<br /><br />
[ 2 ] <b>Compilation is the same as for demo 1</b>

```
// main edk2 folder, go here first
cd edk2

// setup environemnt - read guide for edk2 installation and configuration which I wrote here
edksetup.bat

// maybe you need to clean directory in Build folder for MdeModulePkg in Conf/target.txt if you have some errors in compilation
build cleanall

// compile via .inf just like that
build -p MdeModulePkg/MdeModulePkg.dsc -m MdeModulePkg/Application/HelloWorld/HelloWorld.inf
```

[ 3 ] <b>Some information about files in this folder</b> <br /><br />
HelloWorld.efi - this is file from screenshot running on emulator. I checked also in "real hardware" running from bios and pendrive from path EFI/Boot/bootx64.efi. And on "real hardware" values for RBP and RSP is A7B033F0 and A7B032E8. For CPUID values is the same because VirtualBox not emualte CPU, but use it directly. <br /><br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo2%20-%20print%20registers%20values/132%20-%2006-01-2025%20-%20test%20demo%202%20github.png?raw=true)

invd_asm.txt from scratchpad/bootloader_x86/tianocore EDK2/demo2 - print registers values/HelloWorld/ - I modified file from \MdePkg\BaseLib\x64\lndv.nasm . I tried to make "custom" functions and files for this demo but from what I understand it has to be compiled in the project. So I made a "bypass", i.e. external variables like _vartest etc. And it works. I just have to remember some things about assembler. In lndv.nasm I placed this in section .data. And in HelloWorld.c just call via AsmInvd(); like here https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo2%20-%20print%20registers%20values/HelloWorld/HelloWorld.c#L121C3-L121C13 . The image shows where the path to this file is in the window name at the top of notepad++

![dump](https://raw.githubusercontent.com/KarolDuracz/scratchpad/refs/heads/main/bootloader_x86/tianocore%20EDK2/demo2%20-%20print%20registers%20values/HelloWorld/invd_asm_path.bmp)

<br />
<h2>Summary this demo2</h2>
In this demo. Checked errors from demo1. But also shown how to make external variables, pass register values ​​and then display them. Additionally, there is timer1 (_vartest4_timer1) in the loop. Lots of garbage in the comments and in the code. But I'm posting it as is. How to pass external variable and mix nasm and c code I also checked here <br /><br /> https://github.com/KarolDuracz/scratchpad/tree/main/Hello%20World%20Drivers/demo2 but for EDK2 instead of PUBLIC there is global, instead of QWORD there is DQ etc. 

