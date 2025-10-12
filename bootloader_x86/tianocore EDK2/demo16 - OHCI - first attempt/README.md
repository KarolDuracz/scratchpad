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

The rest of the files in the /bt2/ folder simply list USB devices based on the basic protocols in the UEFI specification. These are simple tests I performed in the previous demo USB 6-7, so I won't describe them here.

<h3>OHCI demo</h3>

What I'm writing here applies to this file -> https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo16%20-%20OHCI%20-%20first%20attempt/bt2/helloworld.efi
<br /><br />
and this code -> https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo16%20-%20OHCI%20-%20first%20attempt/soft%20reset%20ohci/HelloWorld.c
<br /><br />
This is the controller reset sequence

```
Detects OHCI PCI host controllers and attempts a minimal, local initialization:

resets controller (soft reset via HcCommandStatus.HCR),

allocates one physical page for HCCA and programs HcHCCA,

clears control/bulk/done heads,

sets the controller to OPERATIONAL,

reads RhDescriptorA and each root-hub port status and prints connected/enabled/powered/low-speed flags.

If you still see HcHCCA == 0 or root hub ports empty after this, possible causes:

The platform firmware (or another driver) prevents direct MMIO access or ownership (Ownership Change feature / PCI ownership). In such cases you may need to set the PCI Command bits (Memory Space / Bus Master) or request ownership clear via OHCI OCR bit handling — that is driver territory.

IOMMU / DMA mapping: the controller may not be able to access the physical pages you allocated. Real OHCI drivers set up DMA mapping so the HCCA and ED/TD structures are accessible to the controller. My demo uses AllocatePages which often works in UEFI, but on some platforms you must ensure the address is within the device's addressable range (no IOMMU) or use platform-specific mapping.

Platform expects other platform-specific initialization (clocks, regulators, power rails) before the controller can come up. That must be done via platform-specific MMIO or ACPI calls.

```

The result of executing this HelloWrold.efi (equivalent to bootx64.efi)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo16%20-%20OHCI%20-%20first%20attempt/images/104%20-%2012-10-2025%20-%20how%20it%20works%20on%20VirtualBox.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo16%20-%20OHCI%20-%20first%20attempt/images/103%20-%2012-10-2025%20-%20te%202%20porty%20pokazaly%20sie%20jako%20aktywne%20CONNECTED.png?raw=true)

