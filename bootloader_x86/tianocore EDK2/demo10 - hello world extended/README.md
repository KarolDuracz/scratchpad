<h2>EDK2 Hello World extended demo for EmulatorPkg to analyze internal structures and functions</h2>

I found an interesting repo that has a list of many lessons https://github.com/Kostr/UEFI-Lessons/tree/master

<h3>File description - what I changed</h3>
Applies to these files (in demo 1 and 2 I also changed HelloWord instead of creating a new module)
https://github.com/tianocore/edk2/tree/master/MdeModulePkg/Application/HelloWorld
<br /><br />
<b>Source</b><br />
HelloWorld.c - https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo10%20-%20hello%20world%20extended/HelloWorld.c#L199 <br />
HelloWorld.inf - https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo10%20-%20hello%20world%20extended/HelloWorld.inf#L21 <br />
<br />
<b>Build</b> <br />

```
build -p MdeModulePkg/MdeModulePkg.dsc -m MdeModulePkg/Application/HelloWorld/HelloWorld.inf
```

<br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo10%20-%20hello%20world%20extended/344%20-%2027-09-2025%20-%20cd.png?raw=true)

<h3>A short explanation</h3>
This is a demo that can be run on EmulatorPkg https://github.com/tianocore/edk2/tree/master/EmulatorPkg. This is a module that emulates the environment, loads the FD file, and performs subsequent boot steps like PEI, DXE, etc. There's also a SHELL there that can handle the FS0:\ filesystem. After entering this path, we can load applications included with the emulator. Among them is HelloWorld, which by default only prints text to the console. BUT, you can also test other EDK2 (UEFI) mechanisms, such as timers, events, etc.
<br /><br />
<b>Descriptions</b><br />
1. https://uefi.org/specs/UEFI/2.10/07_Services_Boot_Services.html#efi-boot-services-createevent
