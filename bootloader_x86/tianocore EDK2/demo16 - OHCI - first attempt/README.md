> [!WARNING]
> I DO NOT RECOMMEND RUN THIS ON REAL HARDWARE. I AM NOT RESPONSIBLE FOR THE CODE AND CONSEQUENCE.

I also tested these demos using Real HW on my Asus laptop. I'm writing this because after running demo, which attempts to initialize controller after a soft reset, and after restart computer and entering Windows, was unable to initialize ETHERNET connection. It had this problem and repeated process again and again, looking at behavior of 
 tray icon on right bottom corner, which tried to establish a connection but failed. Only shutting down computer fixed this error. This means the system and my firmware, which is currently executing in DXE phase of ASUS system, are probably doing something different than the sequence I've demonstrated here. But this is just a preliminary analysis.
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

What I'm writing here applies to this file -> https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo16%20-%20OHCI%20-%20first%20attempt/helloworld.efi
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

The result of executing this HelloWorld.efi (equivalent to bootx64.efi)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo16%20-%20OHCI%20-%20first%20attempt/images/104%20-%2012-10-2025%20-%20how%20it%20works%20on%20VirtualBox.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo16%20-%20OHCI%20-%20first%20attempt/images/103%20-%2012-10-2025%20-%20te%202%20porty%20pokazaly%20sie%20jako%20aktywne%20CONNECTED.png?raw=true)

<h3>The result of running the rest code test for OHCI</h3>

HELLWORLD_OHCI -> probably this code here - > https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo16%20-%20OHCI%20-%20first%20attempt/HelloWorld%20-%20OHCI%20controller%20demo%2012-10-2025%20-%20working.c

```
FS0:\__bin\a10-10-2025bth\bt3\> helloworld_ohci
Scanning 10 PCI devices for USB host controllers...

PCI USB Host Controller found (handle=0x7DFD5698): Class=0x0C Sub=0x03 ProgIf=0x10
  PCI Location: Seg=0 Bus=0 Dev=6 Func=0
  PCI VendorId=0x106B DeviceId=0x003F
  BAR 0x10 = 0x90627000
  BAR 0x14 = 0x00000000
  BAR 0x18 = 0x00000000
  BAR 0x1C = 0x00000000
  BAR 0x20 = 0x00000000
  BAR 0x24 = 0x00000000
  Interrupt Line=0x0A  Pin=0x01
  PCI DevicePath: PciRoot(0x0)/Pci(0x6,0x0)
  Detected OHCI host controller (ProgIf=0x10). Attempting to read OHCI MMIO registers from BAR0...
    HcRevision(0x00) = 0x00000010
    HcControl(0x04) = 0x00000200
    HcCommandStatus(0x08) = 0x00000000
    HcHCCA(0x18) = 0x00000000
  ConnectController returned: Not Found
  EFI_USB2_HC_PROTOCOL not present for this host (or not found under PCI subtree).

PCI USB Host Controller found (handle=0x7DFD4018): Class=0x0C Sub=0x03 ProgIf=0x20
  PCI Location: Seg=0 Bus=0 Dev=11 Func=0
  PCI VendorId=0x8086 DeviceId=0x265C
  BAR 0x10 = 0x90626000
  BAR 0x14 = 0x00000000
  BAR 0x18 = 0x00000000
  BAR 0x1C = 0x00000000
  BAR 0x20 = 0x00000000
  BAR 0x24 = 0x00000000
  Interrupt Line=0x0B  Pin=0x01
  PCI DevicePath: PciRoot(0x0)/Pci(0xB,0x0)
  ConnectController: OK
  Host Capabilities: maxSpeed=2 numP    Port 1: Connected=0  PortStatus=0x00002100
    Port 2: Connected=0  PortStatus=0x00002100
    Port 3: Connected=0  PortStatus=0x00002100
    Port 4: Connected=0  PortStatus=0x00002100
    Port 5: Connected=0  PortStatus=0x00002100
    Port 6: Connected=0  PortStatus=0x00002100
    Port 7: Connected=0  PortStatus=    Port 8: Connected=0  PortStatus=0x00002100
    Port 9: Connected=0  PortStatus=0x00002100
    Port 10: Connected=0  PortStatus=0x00002100
    Port 11: Connected=0  PortStatus=0x00002100
    Port 12: GetRootHubPortStatus failed: Invalid Parameter

Done. Press any key to exit...
```

HELLOWORLD_OHCI_DEMO2 ==> probably this -> https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo16%20-%20OHCI%20-%20first%20attempt/HelloWorld%20-%20OHCI%20demo2%20-%20ale%20bez%20inicjalizacji%20DXE%20.c

```
FS0:\__bin\a10-10-202helloworld_ohci_demo2
Scanning 10 PCI devices for USB host Found 1 EFI_USB2_HC_PROTOCOL handle(s):
  Usb2Hc handle=0x7DFD4018
    DevicePath: PciRoot(0x0)/Pci(0xB,0x0)
    Protocol[0]: 7DF8A0B0-0000-0000-B0D3-F87D00000000
    Protocol[1]: 7DF8D3B0-0000-0000-3069-FD7D00000000
    Protocol[2]: 7DFD6930-0000-0000-3058-CE7E00000000
    Protocol[3]: 7ECE5830-0000-0000-7074-616C78004200

PCI USB Host Controller found  PCI Location: Seg=0 Bus=0 Dev=6 Func=0
  PCI VendorId=0x106B DeviceId=0x003F
  BAR 0x10 = 0x90627000
  BAR 0x14 = 0x00000000
  BAR 0x18 = 0x00000000
  BAR 0x1C = 0x00000000
  BAR 0x20 = 0x00000000
  BAR 0x24 = 0x00000000
  PCI Command: 0x0017 (MemSpace=1 IO=1 BusMaster=1)
  Interrupt Line=0x0A  Pin=0x01
  PCI DevicePath: PciRoot(0x0)/Pci(0x6,0x0)
  Detected OHCI (ProgIf=0x10). Checking BAR0 MMIO and OHCI registers...    BAR0 MMIO base = 0x90627000
    HcRevision(0x00) = 0x00000010
    HcControl(0x04) = 0x00000200
    HcCommandStatus(0x08) = 0x00000000
    HcHCCA(0x18) = 0x00000000
  ConnectController returned: Not Found
  EFI_USB2_HC_PROTOCOL not present for this host (driver likely missing or not bound).
  Suggestion: run on firmware image with OHCI/EHCI/xHCI DXE driver or add the appropriate DXE driver.

PCI USB Host Controller found (handle=0x7DFD4018): Class=0x0C Sub=0x03 ProgIf=0x20
  PCI Location: Seg=0 Bus=0 Dev=11 Func=0
  PCI VendorId=0x8086 DeviceId=0x265C
  BAR 0x10 = 0x90626000
  BAR 0x14 = 0x00000000
  BAR 0x18 = 0x00000000
  BAR 0x1C = 0x00000000
  BAR 0x20 = 0x00000000
  BAR 0x24 = 0x00000000
  PCI Command: 0x0017 (MemSpace=1 IO=1 BusMaster=1)
  Interrupt Line=0x0B  Pin=0x01
  PCI DevicePath: PciRoot(0x0)/Pci(0xB,0x0)
  ConnectController: OK
  Host Capabilities: maxSpeed=2 numPorts=12 64bit=0
    Port 1: Connected=0  PortStatus=0x00002100
    Port 2: Connected=0  PortStatus=0x00002100
    Port 3: Connected=0  PortStatus=0x00002100
    Port 4: Connected=0  PortStatus=0x00002100
    Port 5: Connected=0  PortStatus=    Port 6: Connected=0  PortStatus=0x00002100
    Port 7: Connected=0  PortStatus=0x00002100
    Port 8: Connected=0  PortStatus=0x00002100
    Port 9: Connected=0  PortStatus=0x00002100
    Port 10: Connected=0  PortStatus=0x00002100
    Port 11: Connected=0  PortStatus    Port 12: GetRootHubPortStatus failed: Invalid Parameter

Diagnostics complete. Press any key to exit...
```

HELLOWORLD_SOFT_INIT_OHCI --> probably this -> https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo16%20-%20OHCI%20-%20first%20attempt/soft%20reset%20ohci/HelloWorld.c or this -> https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo16%20-%20OHCI%20-%20first%20attempt/HelloWorld%20-%20OHCI%20initialization%20soft%2012-10-2025.c

```
FS0:\__bin\a10-10-2025bth\bt3\> helloworld_soft_init_ohci                                                                                                               
Scanning PCI devices for USB host controllers...

PCI USB Host Controller found (handle=0x7DFD5698): Class=0x0C Sub=0x03 ProgIf=0x10
  PCI VendorId=0x106B DeviceId=0x003F
  BAR0 (raw) = 0x90627000
  Detected OHCI host controller. Trying minimal init...
    HcRevision = 0x00000010
    HcControl = 0x00000280
    HcCommandStatus = 0x00000000
    HcHCCA = 0x7CF2B000
    Writing HcCommandStatus.HCR to reset co    OHCI: reset complete
    Allocated HCCA at phys 0x7CF15000
    HcControl set to OPERATIONAL (0x00000280)
    Root hub: 12 downstream ports (RhDescriptorA=0x0000020C)
      Port 1: Status=0x00000000  Connected=0  Enabled=0  Powered=0  LowSpeed=0
      Port 2: Status=0x00010101  Connected=1  Enabled=0  Powered=1  LowSpeed=0
      Po      Port 4: Status=0x00000100  Connected=0  Enabled=0  Powered=1  LowSpeed=0
      Port 5: Status=0x00000100  Connected=0  Enabled=0  Powered=1  LowSpeed=0
      Port 6: Status=0x00000100  Connected=0  Enabled=0  Powered=1  LowSpeed=0
      Port 7: Status=0x00000100  Connected=0  Enabled=0  Powered=1  LowSpeed=0
      Port 8: Status=0x00000100  Connected=0  Enabled=0  Powered=1  LowSpeed=0
      Port 9: Status=0x00000100  Connected=0  Enabled=0  Powered=1  LowSpeed=0
      Port 10: Status=0x00000100  Connected=0  Enabled=0  Powered=1  LowSpeed=0
      Port 11: Status=0x00000100  Connected=0  Enabled=0  Powered=1  LowSpeed=0
      Port 12: Status=0x00000100  Connected=0  Enabled=0  Powered=1  LowSpeed=0

PCI USB Host Controller found (handle=0x7DFD4018): Class=0x0C S  PCI VendorId=0x8086 DeviceId=0x265C
  BAR0 (raw) = 0x90626000
  (Non-OHCI host controller; skipping software init)

Done. Press any key to exit...
```

<h2>SUMMARY</h2>
Initially, I wanted to start interacting with Bluetooth. Since everything works on VirtualBox as it does on the host, i.e., Win 8.1. It detects all devices normally. But, as I mentioned above, it didn't work, so I became interested in the tree topology of devices connected to USB controllers. And that's how I came to OHCI. I started trying to search for devices based on OHCI. And that's how this initial demo for this topic was created. But it's just a rough draft. I didn't get anything else out of it that could give me a step forward right away to start interacting with Bluetooth and using protocols for this specification. But it's still something.

<h3>Links</h3>
1. https://www.bluetooth.com/wp-content/uploads/Files/Specification/HTML/Core-60/out/en/architecture,-change-history,-and-conventions/architecture.html <br />
2. https://devicehunt.com/view/type/usb/vendor/13D3/device/3304 <br />
3. https://uefi.org/sites/default/files/resources/2014_UEFI_Plugfest_AMI.pdf
