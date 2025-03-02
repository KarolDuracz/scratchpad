<h2>Intro - This is not detailed description - checkpoint for me only </h2>

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo%209%20-%20build%20OVMF%20for%20QEMU%20x86_64/demo9-pics/output_demo9.gif?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo%209%20-%20build%20OVMF%20for%20QEMU%20x86_64/demo9-pics/output-demo9-2.gif?raw=true)

I tried to approach building OVMF before delving into how it works in detail. Unfortunately, it is not enough to simply build it with a few commands. What did I find in my case? That it does not work as BIOS for Windows PE. I found some information what could be the cause, but ultimately it did not solve the problem at the moment. But Linux 14.04 loads. But by the way I checked the demos that I uploaded here, i.e. PCI demo, USB demo, the first demos with a green background. And it works on "OVMF default". Unfortunately SECURE BOOT does not work. The file Untitled - explain why secure boot version not working.ipynb which is here has a saved log from QEMU for both versions. I checked the EIP address. In OUT[56] ​​you can see that it oscillates between 1.06 - 1.05. But the last image OUT[61] at the bottom represents running OVMF with SECURE BOOT ENABLE and as you can see from EIP tracing at some point it jumps from
0x3ea19b35
0xe0000
0xdf3e8
And that's why it doesn't work. I tried setting NVRAM but so far I haven't achieved anything. To build this OVMF using command 

```
build -D SECURE_BOOT_ENABLE
```
I changed lines 633 - 637 only in .dsc file I think. In default was 0 in all lines. But here I change to 0xFFF00000, etc.
https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo%209%20-%20build%20OVMF%20for%20QEMU%20x86_64/OVMF%20with%20SECURE%20BOOT%20ENABLED%20-%20not%20working/OvmfPkgX64.dsc#L633
<br /><br />
<b>So to sum it up. It's not 100% working OVMF but since these demos from this repo work, it's partially useful, if only to run and play with these small demos.</b> But looking at the references below, you can see that preparing the environment for the OS is a more complex topic. <br />
<h2>1. Building OvmfPkg and Compiling</h2>
If you have managed to build the previous demos, now all you need to do is change in Conf/target.txt

```
ACTIVE_PLATFORM    = OvmfPkg/OvmfPkgX64.dsc
TARGET_ARCH  = X64
TOOL_CHAIN_TAG  = VS2019
TARGET = RELEASE
```

https://github.com/tianocore/tianocore.github.io/wiki/How-to-build-OVMF
<br /><br />
And then, the same as previous

```
edksetup // setup environment
build // build project
```

If there is an error while building the project as in the picture below, then you need to fix ASL and craete folder C:\ASL . As you can see the compiler looks for this path and if there is neither this path nor files inside then it will throw an error in this place while building OvmfPkg. Just download from this link iasl-win-20240927.zip file and extract to this folder. <br />

https://www.intel.com/content/www/us/en/developer/topic-technology/open/acpica/download.html <br />
https://github.com/user-attachments/files/17171016/iasl-win-20240927.zip 

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo%209%20-%20build%20OVMF%20for%20QEMU%20x86_64/demo9-pics/jesli%20jest%20taki%20blad%20potrzeba%20ASL.png?raw=true)

<hr>
<h2>2. First launch</h2>

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo%209%20-%20build%20OVMF%20for%20QEMU%20x86_64/demo9-pics/pierwsze%20uruchomienie%20po%20zbudowaniu.png?raw=true)

<h2>3. Try demo 2</h2>

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo%209%20-%20build%20OVMF%20for%20QEMU%20x86_64/demo9-pics/220%20-%2027-02-2025%20-%20to%20bylo%20demo%202%20chyba.png?raw=true)

This is base command for run OVMF (I used this here) . I boot with the CDROM parameter and Linux to have access to grub, then I go to UiApp. And with an active USB device which is physically my Win PE pendrive. This is for SHELL. Without it you can only see BLK0 as in the picture above from the first run.

```
qemu-system-x86_64 -L . -bios /share/OVMF.fd -device qemu-xhci,id=xhci -drive if=none,id=usbdisk,file="\\.\PHYSICALDRIVE1",format=raw -cdrom "C:\Users\kdhome\Documents\ImageISO\ubuntu-14.04.6-desktop-amd64.iso" -m 1024 -device usb-storage,drive=usbdisk
```

The command I used to boot with Secure Boot. The Untitled - explain why secure boot version not working.ipynb file and the last 2 graphs show what happens with EIP right after boot. So I probably misconfigured NVRAM. But this is a more complex topic.
```
qemu-system-x86_64 -L . -drive if=pflash,format=raw,unit=0,file=share/ovmf-nvram/OVMF_CODE.fd,readonly=on -drive if=pflash,format=raw,unit=1,file=share/ovmf-nvram/OVMF_VARS.fd -device qemu-xhci,id=xhci -drive if=none,id=usbdisk,file="\\.\PHYSICALDRIVE1",format=raw -cdrom "C:\Users\kdhome\Documents\ImageISO\ubuntu-14.04.6-desktop-amd64.iso" -m 1024 -device usb-storage,drive=usbdisk
```

To save logs to file (don't print on terminal)

```
qemu-system-x86_64 -d int -D debug.log > full.log 2>&1
```

I uploaded some pictures
https://github.com/KarolDuracz/scratchpad/tree/main/bootloader_x86/tianocore%20EDK2/demo%209%20-%20build%20OVMF%20for%20QEMU%20x86_64/demo9-pics
<br /><br />
<h2>4. Summary</h2>
At first I didn't want to write anything, then I made notes, and then I wanted to write much more about it. But at the end I decided that it doesn't work anyway. So it's a shame to produce myself. But I think that what is in the "OVMF default" folder and the command "qemu-system-x86_64 -L . -bios /share/OVMF.fd -device qemu-xhci,id=xhci -drive if=none,id=usbdisk,file="\\.\PHYSICALDRIVE1",format=raw -cdrom "C:\Users\kdhome\Documents\ImageISO\ubuntu-14.04.6-desktop-amd64.iso" -m 1024 -device usb-storage,drive=usbdisk" should work for you on QEMU too. You can try to load this BIOS as I showed here.
<hr>
REFERENCES: <br />
https://www.qemu.org/docs/master/system/i386/pc.html <br />
https://github.com/tianocore/tianocore.github.io/wiki/How-to-Enable-Security <br />
https://superuser.com/questions/1660806/how-to-install-a-windows-guest-in-qemu-kvm-with-secure-boot-enabled <br />
https://projectacrn.github.io/latest/tutorials/waag-secure-boot.html <br />
https://en.opensuse.org/openSUSE:UEFI_Secure_boot_using_qemu-kvm <br />
https://tianocore-docs.github.io/edk2-MinimumPlatformSpecification/draft/7_stage_5_security_enable/75_configuration.html <br />
https://tianocore-docs.github.io/edk2-MinimumPlatformSpecification/draft/6_stage_4_boot_to_os/#6-stage-4-boot-to-os <br />
https://uefi.org/specs/PI/1.8/V3_Design_Discussion.html <br />
https://uefi.org/specs/PI/1.8/V3_Code_Definitions.html

<hr>
<h2>5. Simple test if it works</h2>

1. QEMU is needed - https://www.qemu.org/download/ <br />
2. Download OVMF from this repo - https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo%209%20-%20build%20OVMF%20for%20QEMU%20x86_64/OVMF%20default/OVMF.fd<br />
3. Copy OVMF.fd to some folder, it can be the main folder where qemu is installed.
4. Run virtual machine via this command ( in this case with 512 MB ram -m parameter )

```
qemu-system-x86_64 -L . -bios OVMF.fd -m 512
```
5. This does jump to shell, just like you see on the screen from "2. first launch".
6. Type "exit" on the shell and press ENTER. will jump to UiApp

![qemu](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo%209%20-%20build%20OVMF%20for%20QEMU%20x86_64/demo9-pics/5%20-%20how%20to%20use%20this%20demo%20on%20qemu.png?raw=true)
