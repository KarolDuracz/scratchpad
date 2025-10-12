> [!WARNING]
> I DO NOT RECOMMEND RUN THIS ON REAL HARDWARE. I AM NOT RESPONSIBLE FOR THE CODE AND CONSEQUENCE.

I also tested these demos using Real HW on my Asus laptop. I'm writing this because after running demo, which attempts to initialize controller after a soft reset—simply resetting computer after entering Windows, was unable to initialize ETHERNET connection. It had this problem and repeated process again and again, looking at the behavior of the tray icon on right bottom corner, which tried to establish a connection but failed. Only shutting down computer fixed this error. This means the system and my firmware, which is currently executing in DXE phase of ASUS system, are probably doing something different than the sequence I've demonstrated here. But this is just a preliminary analysis.
<br /><br />
This is a screenshot of the execution of this SOFT INITIALIZATION which you can see below how it works on VirtualBox on the screenshot. <b>In short, program did not work as shown on virtual box.</b> - This is the result of running the program bootx64.efi that is placed here. This is also helloworld.c from this folder, you can see it in the screenshot below that after running this helloworld.efi on VirtulBox, the result is similar.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo16%20-%20OHCI%20-%20first%20attempt/images/1760276157194.jpg?raw=true)

<h3>Okay, so from the beginning. What was the purpose of this demo?</h3>

The goal was to find a Bluetooth device. Even though UEFI Specs lists protocols, I couldn't get any of them to work. I tried various things, at least 20 demos that used things with and without L2Cap. And NOTHING WORKED. So I proceeded to analyze the device topology directly from Windows. <br /><br />
Bluetooth protocol specs - https://uefi.org/specs/UEFI/2.9_A/26_Network_Protocols_Bluetooth.html<br />
USD protocol specs - https://uefi.org/specs/UEFI/2.9_A/17_Protocols_USB_Support.html
<br /><br />
I'm probably doing something wrong, so I gave up and started looking for the tree in which this device is located. And that's how I came across the topic of OHCI.

<h3>Ok, first thing to explain</h3>

In this demo I ran this code - /bt2/helloworld.efi -> https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo16%20-%20OHCI%20-%20first%20attempt/bt2/helloworld.efi
<br /><br />
What's important to note is that it detected the two devices you can see in Windows (Visual Box) on the left. Below that is 0x265C, and above that is OHCI, which has the ID 0x003F. In Windows I didn't show the ID of the second device in path here in Device Manager, but you can see that one ID matches what the shell shows.
<br /><br />
VENDOR and IDs from the source code for comparison<br />
virtualbox/src/VBox/Devices/USB/DevEHCI.cpp - https://github.com/VirtualBox/virtualbox/blob/main/src/VBox/Devices/USB/DevEHCI.cpp#L4905 <br />
virtualbox/src/VBox/Devices/USB/DevOHCI.cpp - https://github.com/VirtualBox/virtualbox/blob/main/src/VBox/Devices/USB/DevOHCI.cpp#L6002 <br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo16%20-%20OHCI%20-%20first%20attempt/images/image3.png?raw=true)

Eemulated device is connected here, meaning it's switching from the host device. I can then see the same vendor and Bluetooth device ID in VirtualBox.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo16%20-%20OHCI%20-%20first%20attempt/images/image1.png?raw=true)

This is what it looks like after disconnecting from VirtualBox. Device returns to the list on the right.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo16%20-%20OHCI%20-%20first%20attempt/images/image2.png?raw=true)

