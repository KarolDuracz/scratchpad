> [!IMPORTANT]
> You can see the "loadshell" and "setbuf" commands in screenshots. Loadshell was supposed to load shell.efi file from \EFI\Boot\. I simply copied compiled shell.efi from EmulatorPkg, but it doesn't work on real HW ( it starts but does not show SHELL, program hangs on the bootx64 loading text ). Setbuf is there because it allows to "scroll" text, but it repaints text N lines back again after pressing LEFT SHIFT + PageUp / PageDown. But then it writes how many lines have been shown. This is an error. I DO NOT RECOMMEND USING THESE COMMANDS. THEY NEED IMPROVEMENT. This demo is mainly for using --> loadimg from myApps.


<h3>shell equivalent - minimal version</h3>

There's only one reason I'm putting it in the demo, because it allows to create a custom folder for applications that can be launched directly from this minimalist shell, like the HelloWorld.efi example. So, in the previous demos, if I wanted to run something on physical hardware (ASUS), I had to rename it to bootx64.efi and put it in the \EFI\Boot\ directory on the flash drive. Here, I have this minimalist shell as bootx64.efi and can drop the rest of the apps directly into the myApps folder, so I can then load them from that path.
<br /><br />
Scenario looks like this: as in the previous demo, I enter the BIOS, then the pendrive and it automatically searches for bootx64.efi (and this is the minimalist shell)

<h3>File list</h3>

bootx64.efi -> file with this "shell" which is on \EFI\Boot\ <br />
HelloWorld.c -> source code <br />
HelloWorld.efi -> This is the same as bootx64.efi <br />
myApps\HelloWorld.efi -> This is a demo from the beginning. It's in the first demo, but I'm uploading it again. It should work for you on QEMU.
<br /><br />
How it looks on physical (real) HW 

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo17%20-%20shell%20equivalent%20-%20minimal%20version/images/real%20hw%20asus.png?raw=true)

Virtual Box

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo17%20-%20shell%20equivalent%20-%20minimal%20version/images/virtualbox.png?raw=true)

qemu, after changing the background color -> gop bg AA77FF

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo17%20-%20shell%20equivalent%20-%20minimal%20version/images/qemu.png?raw=true)

qemu, loadimg command, waits for the entry "yes" and ENTER

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo17%20-%20shell%20equivalent%20-%20minimal%20version/images/qemu2.png?raw=true)

qemu, after entering "yes" and pressing ENTER, the program runs if it was found and there are no errors

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo17%20-%20shell%20equivalent%20-%20minimal%20version/images/qemu3.png?raw=true)

