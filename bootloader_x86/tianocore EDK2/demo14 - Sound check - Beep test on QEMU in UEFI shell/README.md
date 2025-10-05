> [!WARNING]
> [1] Update 05-10-2025 - There's a bug, but I don't know exactly where. Sometimes this BEEP code doesn't work. It just freezes at a certain point. For example, helloworld_beep.efi. For example, in a scenario where you first run the diagnostic 00 05 00 test, then run helloworld.efi and the program freezes. YOU NEED TO RESET THE QEMU SYSTEM. And enter the SHELL again. You might need to do this 2-3 times without the diagnostic test, just run one of the three available .efi files that generates the BEEP. There's helloworld.efi, helloworld_beep.efi, and helloworld_first.efi. Keep it in mind. This doesn't mean the code I posted isn't working. It's just that something isn't quite right. But I don't know what yet. <br />
> [2] The second error is that after executing 1x BEEP you need to reset the system because running this code again does not work. <br />
> Just keep it in mind. That doesn't mean it doesn't work at all. <br /><br />
> When its freeze on lines <br />
>         Preparing tiny AC'97 PCM OUT DMA test (single BDL entry) <br />
>         Allocating audio buffer: 22050 samples (44100 bytes), 11 pages <<<<<<<<<<<< here <br />
> try to reset system by clicking CTRL + ALT + G to release the mouse and on "Machine > Reset" menu in the main QEMU window <br />
> sometimes it freezes at the end of the code when waiting for a key, after the beep.

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

VENDOR and device ID from shell indicated for this sound card that this device is emulated - SigmaTel Intel r AC'97 Audio Controller - https://sunsite.icm.edu.pl/pub/linux/alsa/manuals/sigmatel/9721spec.PDF  - it still needs to be checked. But to have some preliminary insight.

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

The first thing you should know is how I set up recording in Windows 8.1 to record from the system <br />
1. I locked microphone <br />
2. I set stereo mix as default <br />
3. I recorded using the "Sound Recorder" app <br />
4. I saved it to the "beep_test.wma" file <br />
5. Around 8-9 am, a beep sound appears in the .wma file <br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo14%20-%20Sound%20check%20-%20Beep%20test%20on%20QEMU%20in%20UEFI%20shell/images/windows%208%20sound%20settings.png?raw=true)

here is a comparison of what the PCI command sees on qemu with and without an audio device in the command.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo14%20-%20Sound%20check%20-%20Beep%20test%20on%20QEMU%20in%20UEFI%20shell/images/13%20-%2005-10-2025%20-%20qemu%20vs%20qemu%20z%20audio%20device%20na%20PCI.png?raw=true)

PUTTY setup to have a listing on port 4444

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo14%20-%20Sound%20check%20-%20Beep%20test%20on%20QEMU%20in%20UEFI%20shell/images/putty%20launch.png?raw=true)

thanks to this, I can copy text content to the clipboard using putty menu

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo14%20-%20Sound%20check%20-%20Beep%20test%20on%20QEMU%20in%20UEFI%20shell/images/step%206%20-%20diganostic%20test%20on%20pci%20device%2000%2005%2000.png?raw=true)

Ok, time for a test. The first thing I check is "PCI"

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo14%20-%20Sound%20check%20-%20Beep%20test%20on%20QEMU%20in%20UEFI%20shell/images/step%204%20-%20pci%20devices%20list.png?raw=true)

header for 00 05 00 device. And this is exactly what you can see in the image above from helloworld_pci00-05-00_diagnostic.efi which checks access to these IO registers.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo14%20-%20Sound%20check%20-%20Beep%20test%20on%20QEMU%20in%20UEFI%20shell/images/step%205%20-%20read%20pci%20header%20for%2000%2005%2000.png?raw=true)

confirmation that everything compiles for me with this piece of code

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo14%20-%20Sound%20check%20-%20Beep%20test%20on%20QEMU%20in%20UEFI%20shell/images/1%20-%20original%20hello%20world%20-%20first%20test%20-%20passed.png?raw=true)

After the diagnostic test (to check if it works according to my demo) I run the first test

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo14%20-%20Sound%20check%20-%20Beep%20test%20on%20QEMU%20in%20UEFI%20shell/images/step%202.png?raw=true)

As a result, you can see what is in the terminal and in the meantime a BEEP sound appears

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo14%20-%20Sound%20check%20-%20Beep%20test%20on%20QEMU%20in%20UEFI%20shell/images/step%203%20-%20run%20basic%20beep%20test.png?raw=true)

Another test 

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo14%20-%20Sound%20check%20-%20Beep%20test%20on%20QEMU%20in%20UEFI%20shell/images/step%207%20-%20hello%20world%20beep%20second%20test.png?raw=true)

But it also uses DMA. Sometimes it locks up. You have to reset QEMU.

<h3>Summary</h3>

This is a deep topic. After generating the BEEP, next comes the attempt to play the melody. This works in mono. I won't go into details for now. Maybe next time I manage to play the "Jingle Bells" melody ;p

<h3>Virtual Box test</h3>

In the previous demo, I also tried running the test on VirtualBox 7. But this time it's not a straight-forward test, because it looks like a different device. I haven't checked it thoroughly yet, but the "pci" listing shows a different ID. So the registers are probably different too, I think.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo14%20-%20Sound%20check%20-%20Beep%20test%20on%20QEMU%20in%20UEFI%20shell/virtualbox%207%20test%20and%20pins%20datasheet/virtualbox_pci_find_audio_device.png?raw=true)

It looks like this is an audio device. I've added a few more screenshots here to give you a preview of what it looks like in VirtualBox. https://github.com/KarolDuracz/scratchpad/tree/main/bootloader_x86/tianocore%20EDK2/demo14%20-%20Sound%20check%20-%20Beep%20test%20on%20QEMU%20in%20UEFI%20shell/virtualbox%207%20test%20and%20pins%20datasheet  - There are also a few images with pins, but that's just a preliminary view. Don't pay attention to that for now. That needs to be checked carefully next time. But the PCI 00 05 00 diagnostic test works, because it's also on this line in VirtualBox. So, as you can see in screenshot there, something is printed. File "virtualbox_diagnostic pci 00 05 00.png".

<h3>LINKS</h3>
1. QEMU sound cards list - https://computernewb.com/wiki/QEMU/Devices/Sound_cards - AC97 shows Intel(r) 82801AA AC'97 (SigmaTel STAC9750 codec) <br />
2. SigmaTel Intel r AC'97 Audio Controller datasheed - https://sunsite.icm.edu.pl/pub/linux/alsa/manuals/sigmatel/9721spec.PDF<br />
3. High Definition Audio Specification Revision 1.0a June 17, 2010 - https://www.intel.com/content/dam/www/public/us/en/documents/product-specifications/high-definition-audio-specification.pdf - I see a lot of changes from '04 on the first pages describing this revision. But documentation will probably match registers in Virtual Box, I think.<br />
4. https://www.virtualbox.org/browser/vbox/trunk/src/VBox/Devices/Graphics/ - source code. This is VGA. But here is a implementation and source code for Audio Card also I think.
<br /><br />
5. qemu/hw/audio/ac97.c --> https://github.com/qemu/qemu/blob/master/hw/audio/ac97.c

<h3>Real HW test?</h3>

On my ASUS laptop? This is dump from registry > SYSTEM > CurrentControlSet > Enum > HDAUDIO ( tree showing 2 devices, 2 items )

```
HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Enum\HDAUDIO\FUNC_01&VEN_10EC&DEV_0269&SUBSYS_10431AA3&REV_1001 <<<< DeviceDesc : Realtek High Definition Audio
HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Enum\HDAUDIO\FUNC_01&VEN_8086&DEV_2805&SUBSYS_80860101&REV_1000
```

These are important two values 10EC&DEV_0269 and 8086&DEV_2805. One of them shows that it is Realtek Semiconductor Corp. https://catalog.update.microsoft.com/Search.aspx?q=VEN_10EC%26DEV_0269 - Correct. <br />
So I don't even run it to avoid damaging something in the system.

