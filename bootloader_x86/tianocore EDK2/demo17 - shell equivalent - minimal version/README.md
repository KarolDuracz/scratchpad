<h3>shell equivalent - minimal version</h3>

There's only one reason I'm puting it in the demo, because it allows to create a custom folder for applications that can be launched directly from this minimalist shell, like the HelloWorld.efi example. So, in the previous demos, if I wanted to run something on physical hardware (ASUS), I had to rename it to bootx64.efi and put it in the \EFI\Boot\ directory on the flash drive. Here, I have this minimalist shell as bootx64.efi and can drop the rest of the apps directly into the myApps folder, so I can then load them from that path.

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

