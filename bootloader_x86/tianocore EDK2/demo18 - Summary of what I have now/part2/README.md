I apologize for my poor English.
<br /><br />
// WARNING - if a file doesn't compile, add // or some text as a comment somewhere to update the date of file.
// the point is that if you copy this file to disk and run "build" but the date is different, it might not build a "fresh" helloworld.efi
// even though you won't see any errors. So each file needs to be slightly modified to be up-to-date (current time and date) before the "build" command.
<br /><br />
Compare what's in the code given to the Print function or the wrappers for that function that will print something to the console. Simply press CTRL + F, type "Print," and you'll get all the lines with what's printed to the console. This way, you can compare the code with what's in the logs. I won't go into lengthy descriptions here. Just open the code and log and see where everything is displayed.
<br /><br />
Everything is built on HelloWorld https://github.com/tianocore/edk2/tree/master/MdeModulePkg/Application/HelloWorld

```
// preparing the environment and variables
// cd C:\Users\kdhome\Documents\progs\edk2_win81\edk2\
edksetup

// building the project (EmulatorPkg) - from this I copy the helloworld.efi file to the flash drive
build
```

<b>copy edk2 to usb.bat</b> - This is a BAT script that simply copies the helloworld.efi file after building the EmulatorPkg project to a pendrive in myApps, from which it then runs this minimalistic shell<br /><br />
<b>index_v3.html</b> - A small HTML program that decodes register bits. A little help with reading register values ​​from the readout and documentation.

index_v3.html

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo18%20-%20Summary%20of%20what%20I%20have%20now/part2/175%20-%2016-10-2025%20-%20cd.png?raw=true)

<h3>//////////////////////////////////////////////////////// DEMO 0</h3>

// There's no chat for this demo. Only the code.

No description for "demo3 - diag ver 1"
No description for "demo2 - diagnose for this log see image" - there's a LOG.txt file inside, so it's a potential log for this demo, but this initial demo was before what's in the image.

It all started with an attempt to initially diagnose the PCI configuration and basic registers. So, I started analyzing the documentation to see what each register and its set value tell me.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo18%20-%20Summary%20of%20what%20I%20have%20now/part2/1760358308777.jpg?raw=true)

<h3>//////////////////////////////////////////////////////// DEMO 1</h3>

// It would be better if I just added links to these chats. That would be the best solution.

https://chatgpt.com/share/68f3fb8f-02dc-8000-9570-fcb651dbf1c6

Demo 1 focuses on the EnumerateViaUsbIo() function. I started looking at the parameters of the functions I use, such as 7.3.15. EFI_BOOT_SERVICES.LocateHandleBuffer()
(https://uefi.org/specs/UEFI/2.10/07_Services_Boot_Services.html#id32) and there's a parameter IN EFI_LOCATE_SEARCH_TYPE SearchType.

Looking at the structure, there are three modes. I was still only using ByProtocol. So I started wondering if that might be the reason I wasn't detecting something in these diagnostic tests.

```
//******************************************************
// EFI_LOCATE_SEARCH_TYPE
//******************************************************
typedef enum {
   AllHandles,
   ByRegisterNotify,
   ByProtocol
  } EFI_LOCATE_SEARCH_TYPE;
```

Did changing to AllHandles help? I'm not 100% sure, but I doubt it. But I'll note that I tried it.

So, this demo returned all installed protocols. This is the VirtualBox log. I also tested it with real hw. I don't remember exactly how many protocols there are, but it's still a pretty long list.

<h3>//////////////////////////////////////////////////////// DEMO 2</h3>

https://chatgpt.com/share/68f13307-7b28-8000-8fdb-c56ca702de88

There are five folders here. Each contains a "log.txt" file, so you can see what the potential results of running this on the VIRTUAL BOX are.

But demo 5 is important here, because it's here that we managed to read the exact name of the USB flash drive in LOG7.txt.

```
  --- End diagnose for port 2 ---
  USBCMD changed to 0x00000031 ??? driver may hav--- Inspect PCI devices for Vendor=0x106B Device=0x003F ---
Found matching PCI handle: handle=7DFD5698 (index=6)
  -> PCI location: Segment=0 Bus=00 Dev=06 Func=0
    Vendor/Device = 0x003F106B (Vendor=0x106B Device=0x003F)
    PCI Command = 0x0017 (IO=1 MEM=1    PCI BAR0 = 0x90627000
    PCI config (0x00..0x3C):
      [00] = 0x003F106B
      [04] = 0x00100017
      [08] = 0x0C031000
      [0C] = 0x00000000
      [10] = 0x90627000
      [14] = 0x00000000
      [18] = 0x00000000
      [1C] = 0x00000000
      [20] = 0x00000000
      [24] = 0x00000000
      [28] = 0x00000000
      [2C] = 0x00000000
      [30] = 0x00000000
      [34] = 0x00000000
      [38] = 0x00000000
      [3C] = 0x0000010A
  Scanning 1 EFI_USB_IO_PROTOCOL handles for device path PCI node ma  Found bulk endpoints: OUT=0x01 IN=0x82
  UsbBulkTransfer(CBW OUT): Status = 0x0
  UsbBulkTransfer(INQUIRY DATA-IN): Status = 0x0
    INQUIRY returned 36 bytes
    INQUIRY data (len=36):
0000: 00 80 02 02 1F 00 00 00 47 65 66769 63 20
0010: 46 6C 61 73 68 20 44 69 73 6B 20 20 20 20 20 20
0020: 38 2E 30 37
    INQUIRY: Vendor="Generic " Product="Flash Disk      " Rev="8.07" 		// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  UsbBulkTransfer(CSW IN): Status = 0x0
    CSW status = 0x00 DataResidue=0x00000000
    SCSI INQUIRY completed successfully (no writes performed)
--- End Inspect ---
```

<h3>//////////////////////////////////////////////////////// DEMO 3</h3>

https://chatgpt.com/share/68f404e3-5024-8000-bdc1-de02800b6bd6

This is the longest code I wrote about in part 1, those 4,000 lines. Here's the VirtualBox log, too, but it was also tested with real hardware, because without it, I wouldn't have been able to check
whether it was possible to interact with this USB device, i.e., whether it was writable.

THIS LOG IS WORTH REVIEWING. AS IS THE CODE. But I won't go into detail about it for now. I simply recommend reviewing this code, as there's probably more to be learned
in the next demos.

<h3>//////////////////////////////////////////////////////// DEMO 4</h3>

Here gpt chat provided the sequence (safe version according to him the device detection sequence starting from PCI and then through the EHCI / OHCI controller)

link to chat 1 -> https://chatgpt.com/share/68f4079c-e470-8000-a33f-3cd56a369b8c

link to chat 2-> https://chatgpt.com/share/68f39238-5464-8000-b1db-0c7797b3bf2a

This is the demo shown in the images from Part 1 on real HW, detecting the device type. Only here is the VirtualBox log. It waits 5 seconds for the device and performs some diagnostic tests.

<h3>//////////////////////////////////////////////////////// DEMO 5 - NETWORK demo</h3>

https://chatgpt.com/share/68f40761-2078-8000-82bf-3548ab2b2d91

Images can be seen in part 1 for real hw and VirtualBox. Here is the code and log.

<h3>//////////////////////////////////////////////////////// DEMO 6 - savelog</h3>

There is an example log that was saved from VirtualBox - log-20251018-031139-0002.txt


