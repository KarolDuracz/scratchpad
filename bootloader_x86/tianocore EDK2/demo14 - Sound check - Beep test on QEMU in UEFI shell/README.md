<h2>Sound check - Beep test on QEMU in UEFI shell</h2>
What was my goal for this demo? First and foremost, find code that would work. Not only would it compile, but it would actually trigger the BEEP sound in the shell. There are a few steps I took before finding the right code with GPT-5 help, but I'll just focus on what I have, and what I managed to get working, AND IT LOOKS LIKE IT WORKS.
<br /><br />
Ok, back to https://github.com/Kostr/UEFI-Lessons/tree/master at the bottom is a list of interesting links to other repos and sites about UEFI. One of the links is a repo from user fpmurphy. Unfortunately, the page where he supposedly described what this demo is all about has disapeard. But the GitHub repo remains https://github.com/fpmurphy/UEFI-Utilities-2019/tree/master/MyApps/Beep. Unfortunately, this demo, despite having an .efi file, DOESN'T WORK FOR ME on QEMU. But that's where I started, to find code that would try to make Beep in the same way. (...) There are many topics around, so this demo should be treated as an attempt to find code that can be run as a first test and find something that works. Without delving into the details of how it works now (I'll come back to this later).

<h3>What's included in this demo?</h3>
1. HelloWorld.c - https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo14%20-%20Sound%20check%20-%20Beep%20test%20on%20QEMU%20in%20UEFI%20shell/HelloWorld.c#L236 - First of all, I uploaded "fresh" helloworld.c and INF files from the default EDK2 folder, so as not to mess with timers, etc. from the 10-13 demo again https://github.com/tianocore/edk2/tree/master/MdeModulePkg/Application/HelloWorld. Around line 236, the code that generates sound starts. This is probably the file that generated helloworld_beep.efi <br />
2. helloworld_beep.efi - file that is ready for testing, you can see in the image that I am running it. <br />
3. helloworld_pci00-05-00_diagnostic.efi - diagnostic test for PCI device on 00 05 00.  <br />
4. HelloWorld - pci 00 05 00 diagnostic 05-10-2025.c - https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo14%20-%20Sound%20check%20-%20Beep%20test%20on%20QEMU%20in%20UEFI%20shell/HelloWorld%20-%20pci%2000%2005%2000%20diagnostic%2005-10-2025.c - This is the code that produces helloworld_pci00-05-00_diagnostic.efi. You just have to paste it into helloworld.c and compile helloworld.efi. But I changed the names later. <br />
5. /helloworld_first_test_passed/HelloWorld.c - https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo14%20-%20Sound%20check%20-%20Beep%20test%20on%20QEMU%20in%20UEFI%20shell/helloworld_first_test_passed/HelloWorld.c#L214 - With this loop starting at line 214, this was the first demo I managed to run and generate BEEP. So consider this as the base code. <br />
6. helloworld_first.efi - the first demo that worked for me <br />
 <br /> 
7. <b>beep_test.wma</b> -  I recorded the file to confirm that it actually generates the BEEP sound in the shell. First, Google Translate says, then it counts down 3, 2, 1. Around 8-9, you hear BEEP sound. Then I run the helloworld_beep.efi script, and then the translator says "done." - https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo14%20-%20Sound%20check%20-%20Beep%20test%20on%20QEMU%20in%20UEFI%20shell/beep_test.wma

<h3>A few commands</h3>

The first thing I did was look for some devices in qemu that I have installed

```
C:\Program Files\qemu>qemu-system-x86_64 -device AC97,help
AC97 options:
  acpi-index=<uint32>    -  (default: 0)
  addr=<int32>           - Slot and optional function number, example: 06.0 or 06 (default: -1)
  audiodev=<str>         - ID of an audiodev to use as a backend
  failover_pair_id=<str>
  multifunction=<bool>   - on/off (default: false)
  rombar=<uint32>        -  (default: 1)
  romfile=<str>
  romsize=<uint32>       -  (default: 4294967295)
  x-pcie-extcap-init=<bool> - on/off (default: true)
  x-pcie-lnksta-dllla=<bool> - on/off (default: true)
```

This is a script that adds the device that is visible in the listing from the "pci" command

```
qemu-system-x86_64 \
  -L . -bios /share/OVMF.fd \
  -device qemu-xhci,id=xhci \
  -drive if=none,id=usbdisk,file="\\.\PHYSICALDRIVE1",format=raw \
  -cdrom "C:\Users\kdhome\Documents\ImageISO\ubuntu-14.04.6-desktop-amd64.iso" \
  -m 1024 -device usb-storage,drive=usbdisk \
  -audiodev dsound,id=snd0,latency=20000 \
  -device AC97,audiodev=snd0 \
  -machine pcspk-audiodev=snd0
```

But for these demos I run QEMU from the command line because I also wanted to have PUTTY to be able to copy the contents of SHELL + serial tcp:127.0.0.1:4444,server,nowait

```
qemu-system-x86_64 -L . -bios /share/OVMF.fd -device qemu-xhci,id=xhci -drive if=none,id=usbdisk,file="\\.\PHYSICALDRIVE1",format=raw -cdrom "C:\Users\kdhome\Documents\ImageISO\ubuntu-14.04.6-desktop-amd64.iso" -m 1024 -device usb-storage,drive=usbdisk -audiodev dsound,id=snd0,latency=20000 -device AC97,audiodev=snd0 -machine pcspk-audiodev=snd0 -serial tcp:127.0.0.1:4444,server,nowait
```

VENDOR and device ID from shell indicated for this sound card that this device is emulated - SigmaTel Intel r AC'97 Audio Controller - https://sunsite.icm.edu.pl/pub/linux/alsa/manuals/sigmatel/9721spec.PDF

```
FS1:\__bin\a28-09-2025\beep\> pci
   Seg  Bus  Dev  Func
   ---  ---  ---  ----
    00   00   00    00 ==> Bridge Device - Host/PCI bridge
             Vendor 8086 Device 1237 Prog Interface 0
    00   00   01    00 ==> Bridge Device - PCI/ISA bridge
             Vendor 8086 Device 7000 Prog Interface 0
    00   00   01    01 ==> Mass Storage Controller - IDE controller
             Vendor 8086 Device 7010 Prog Interface 80
    00   00   01    03 ==> Bridge Device - Other bridge type
             Vendor 8086 Device 7113 Prog Interface 0
    00   00   02    00 ==> Display Controller - VGA/8514 controller
             Vendor 1234 Device 1111 Prog Interface 0
    00   00   03    00 ==> Network Controller - Ethernet controller
             Vendor 8086 Device 100E Prog Interface 0
    00   00   04    00 ==> Serial Bus Controllers - USB
             Vendor 1B36 Device 000D Prog Interface 30
    00   00   05    00 ==> Multimedia Device - Audio device  <<<<<<<<<<<<<<<<<<<<<<<< HERE !!!
             Vendor 8086 Device 2415 Prog Interface 0        <<<<<<<<<<<<<<<<<<<<<<<< HERE !!!
FS1:\__bin\a28-09-2025\beep\> pci 00 05 00
  PCI Segment 00 Bus 00 Device 05 Func 00 [EFI 0000050000]
  00000000: 86 80 15 24 07 00 80 02-01 00 01 04 00 00 00 00  *...$............*
  00000010: 01 C0 00 00 01 C4 00 00-00 00 00 00 00 00 00 00  *................*
  00000020: 00 00 00 00 00 00 00 00-00 00 00 00 F4 1A 00 11  *................*
  00000030: 00 00 00 00 00 00 00 00-00 00 00 00 0A 01 00 00  *................*

  00000040: 00 00 00 00 00 00 00 00-00 00 00 00 00 00 00 00  *................*
  00000050: 00 00 00 00 00 00 00 00-00 00 00 00 00 00 00 00  *................*
  00000060: 00 00 00 00 00 00 00 00-00 00 00 00 00 00 00 00  *................*
  00000070: 00 00 00 00 00 00 00 00-00 00 00 00 00 00 00 00  *................*
  00000080: 00 00 00 00 00 00 00 00-00 00 00 00 00 00 00 00  *................*
  00000090: 00 00 00 00 00 00 00 00-00 00 00 00 00 00 00 00  *................*
  000000A0: 00 00 00 00 00 00 00 00-00 00 00 00 00 00 00 00  *................*
  000000B0: 00 00 00 00 00 00 00 00-00 00 00 00 00 00 00 00  *................*
  000000C0: 00 00 00 00 00 00 00 00-00 00 00 00 00 00 00 00  *................*
  000000D0: 00 00 00 00 00 00 00 00-00 00 00 00 00 00 00 00  *................*
  000000E0: 00 00 00 00 00 00 00 00-00 00 00 00 00 00 00 00  *................*
  000000F0: 00 00 00 00 00 00 00 00-00 00 00 00 00 00 00 00  *................*
```

DIAGNOSTIC PCI TEST FOR 00 05 00 PCI DEVICE - helloworld_pci00-05-00_diagnostic.efi 

```
DumpPciBar: locating EFI_PCI_IO handles...
Found PCI device at 0:5.0

PCI config (0x00..0x3C):
  +0x00 : 0x24158086
  +0x04 : 0x02800007
  +0x08 : 0x04010001
  +0x0C : 0x00000000
  +0x10 : 0x0000C001
  +0x14 : 0x0000C401
  +0x18 : 0x00000000
  +0x1C : 0x00000000
  +0x20 : 0x00000000
  +0x24 : 0x00000000
  +0x28 : 0x00000000
  +0x2C : 0x11001AF4
  +0x30 : 0x00000000
  +0x34 : 0x00000000
  +0x38 : 0x00000000
  +0x3C : 0x0000010A

PCI Command register = 0x0007
  bit0 = I/O Space (ENABLED)
  bit1 = Memory Space (ENABLED)
  bit2 = Bus Master (ENABLED)

BARs:
  BAR0: I/O space, raw=0x0000C001, base=0x0000C000
    First dword at I/O BAR0 offset 0 = 0xFFFFFFFF
  BAR1: I/O space, raw=0x0000C401, base=0x0000C400
    First dword at I/O BAR1 offset 0 = 0x00000000
  BAR2: MMIO (32-bit), raw=0x00000000, base=0x0000000000000000
    PciIo->Mem.Read failed for BAR2: Unsupported
  BAR3: MMIO (32-bit), raw=0x00000000, base=0x0000000000000000
    PciIo->Mem.Read failed for BAR3: Unsupported
  BAR4: MMIO (32-bit), raw=0x00000000, base=0x0000000000000000
    PciIo->Mem.Read failed for BAR4: Unsupported
  BAR5: MMIO (32-bit), raw=0x00000000, base=0x0000000000000000
    PciIo->Mem.Read failed for BAR5: Unsupported

Done. Press any key to exit.
```

<h3>Ok, some pictures</h3>

