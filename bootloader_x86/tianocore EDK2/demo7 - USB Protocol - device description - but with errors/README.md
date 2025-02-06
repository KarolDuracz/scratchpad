<h2>EFI_USB2_HC_PROTOCOL.ControlTransfer()</h2>
I tried to get data about the device but it requires more effort to understand the USB operation. And the protocol itself. But I managed to dig something up at the end of the day. I AM UPLOADING THE .EFI FILE FOR DEBUGGING PURPOSES ONLY. I DO NOT RECOMMEND RUN IT ON YOUR OWN MACHINE.
<br /><br />
https://uefi.org/specs/UEFI/2.10/17_Protocols_USB_Support.html#efi-usb2-hc-protocol-controltransfer
<br /><br />
I don't know if it works properly, but as you can see it get something that matches the documentation. For example 0x9 from this list
https://www.usb.org/defined-class-codes is equal to HUB. And 8087 is Intel Vendor Id. But this is not working properly. 
<br /><br />
Code in line 413 - https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo7%20-%20USB%20Protocol%20-%20device%20description%20-%20but%20with%20errors/HelloWorld.c#L413 - execution in line 395.
I'll stop here for a moment to read more about this.
<br /><br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo7%20-%20USB%20Protocol%20-%20device%20description%20-%20but%20with%20errors/58%20-%2006-02-2025%20-%20cd.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo7%20-%20USB%20Protocol%20-%20device%20description%20-%20but%20with%20errors/1738882436063.jpg?raw=true)
