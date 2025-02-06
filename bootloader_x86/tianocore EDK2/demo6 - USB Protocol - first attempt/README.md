<h2>USB Protocol</h2>

Finally, something that has interested me for a years - how USB works. First approach to USB Protocol.
<br /><br />
Same as previous demo - HelloWorld.c - source code, HelloWorld.inf - dependencies, HelloWorld.efi - compiled code for this demo6.
<br /><br />
https://uefi.org/specs/UEFI/2.10/17_Protocols_USB_Support.html#id9
<br /><br />

first I checked on Virtualbox, because USB is already supported and emulated. For example, Emulator EDK2 in basic setup don't have this type of device, PCI also etc. Only simple console protocols.<br /><br />
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo6%20-%20USB%20Protocol%20-%20first%20attempt/53%20-%2006-02-2025%20-%20ok%20na%20virtual%20box%20dziala%20teraz%20test%20na%20real%20hw.png?raw=true)

Real hardware. I don't know if the information displayed is correct. But this is what it returned. Something doesn't seem right here for me. In the next demo, needs to display more information to find out what it actually showed me here.<br /><br />
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo6%20-%20USB%20Protocol%20-%20first%20attempt/1738870154525.jpg?raw=true)

Function used for this demo is in line 332. Execution in line 514.
https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo6%20-%20USB%20Protocol%20-%20first%20attempt/HelloWorld.c#L332
<br /><br />
This is an interesting protocol, so I need to look into it further more. Learn more about it. Many different types of devices, controllers, keyboards, mice, SSDs hard disk, pendrives, etc. There are many classes of devices that use this protocol and USB connectors.
<br /><br />
References: <br />
https://www.virtualbox.org/browser/vbox/trunk/src/VBox/Devices/USB <br />
https://en.wikipedia.org/wiki/USB <br />
https://www.usb.org/document-library/usb-20-specification - official USB 2.0 Specification
