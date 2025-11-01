I will skip the long description this time. I'll just show how it works on images, what I changed in these demos.
<br /><br />
<b>1. Minimalist Shell FIXED AND IMPROVED.</b>
<br /><br />
https://github.com/KarolDuracz/scratchpad/tree/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes
<br /><br />
<b>2. Simple text editor for logs ( read only )</b>
<br /><br />
https://github.com/KarolDuracz/scratchpad/tree/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo2%20-%20log%20reader
<br /><br />
<b>3. GOP fun - playing with custom fonts and image loading, simple pointer protocol, etc.</b>
<br /><br />
https://github.com/KarolDuracz/scratchpad/tree/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo3%20-%20gop%20fun
<br /><br /><br />
> [!NOTE]
> How to build demos from source code - I build it on EmulatorPkg ( helloworld ) and copy helloworld.efi

Every demo from 1-19 (up to now) is built on top of HelloWorld.c https://github.com/tianocore/edk2/tree/master/MdeModulePkg/Application/HelloWorld . So I build it with the commands. But you could also build it with EmulatorPkg and copy helloworld.efi ( in demo 18 there is a .bat script that does this ) 

```
// main edk2 folder, go here first
cd edk2

// setup environment - read guide for edk2 installation and configuration which I wrote here
edksetup.bat

// compile via .inf just like that
build -p MdeModulePkg/MdeModulePkg.dsc -m MdeModulePkg/Application/HelloWorld/HelloWorld.inf
```
