<h3>Introduction</h3>
I tried to find something about the operation and detection of devices connected to HDMI. Looking at this manual from 2011 it looks like it must be done 
by EFI_DRIVER_BINDING_PROTOCOL. But this is only a preliminary insight where you can look for it. As here I checked the subject of VGA and what is the Basicdisplay 
Detection sequence in Intel manuals, it already shows that this is not a trivial topic after all <br />
1. https://www.intel.co.uk/content/dam/doc/guide/uefi-driver-graphics-controller-guide.pdf<br />
2. https://uefi.org/specs/UEFI/2.9_A/11_Protocols_UEFI_Driver_Model.html#efi-driver-binding-protocol-supported-protocols-uefi-driver-model<br />
3. https://uefi.org/sites/default/files/resources/UEFI%20Spec%202_6.pdf<br />

<h3>DEMO</h3>
The folders contain a compiled demo that can be started on QME but it is tested on my laptop on the Intel HD 3000 (which can be seen in 2 pictures below). I have already written what I do in the previous demos in this repo but I will write again. I just compile hellowrold.efi and change the file name from helloworld.efi > bootx64.efi. That's all. Then I copy to the USB flash drive to the "\efi\boot\bootx64.efi" folder. I start a computer, enter the BIOS (F2), and as you can see on screenshot, then I choose booting from USB
<br /><br />
1. EDID demo - https://github.com/KarolDuracz/scratchpad/tree/main/bootloader_x86/tianocore%20EDK2/demo12%20-%20GOP%20again/GOP_demo1_edid
<br /><br />
2. Color demo - https://github.com/KarolDuracz/scratchpad/tree/main/bootloader_x86/tianocore%20EDK2/demo12%20-%20GOP%20again/GOP_demo2_color
<br /><br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo12%20-%20GOP%20again/1759071908284.jpg?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo12%20-%20GOP%20again/1759071908294.jpg?raw=true)

<h3>More tests on QEMU - The same as in the GOP_DEMO1_EDID folder bootx64.efi</h3>

Demo from https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo12%20-%20GOP%20again/HelloWorld%20-%20GOP_HDMI_28-09-2025.c - This does not test colors, but is to detect the number of GOP, settings, and show a list of all possible modes and an EDID fragment. Unfortunately, I can't take a good photo from a laptop now, it also works on real hardware and detects these edid in these lines. This is not compiled. It's only as source code.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo12%20-%20GOP%20again/379%20-%2028-09-2025%20-%20probuje%20zrobic%20HDMI%20driver.png?raw=true)

<h3>More tests on QEMU - The same as in the GOP_DEMO2_COLOR folder bootx64.efi</h3>

1. HelloWorld.c - https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo12%20-%20GOP%20again/HelloWorld.c - color test
2. HelloWorld.INF - https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo12%20-%20GOP%20again/HelloWorld.inf - It was necessary to add gefiedddiscoveredprotocolguid
3. helloworld.efi - https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo12%20-%20GOP%20again/HelloWorld.efi - compiled color test that can be started QEMU

The test begins with displaying some information about GOP, but skipped EDID. EDID is only in the first demo. Here is only a color test.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo12%20-%20GOP%20again/test1%20-%20qemu.png?raw=true)

Then he launches the swiction of several tests in the loop for about 10 seconds - Animation of moving to the right of a rectangle with changing colors.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo12%20-%20GOP%20again/test2.png?raw=true)

Then the column test

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo12%20-%20GOP%20again/test3.png?raw=true)

Grid test

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo12%20-%20GOP%20again/test4.png?raw=true)

And at the end he is waiting to press the key to go to Shell or fall into Deadloopcpu, i.e. a loop infinite for safety

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo12%20-%20GOP%20again/test5.png?raw=true)


