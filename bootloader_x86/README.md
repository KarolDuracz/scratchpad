First. From this 3 files .ISO / .BIN works only boot_qemu_x86.bin. But this is raw file from disk. This is not fancy demo. This bootloader.asm and boot_qemu_x86.asm ... by default they were supposed to be bootable via CDROM because they are supposed to be burned to CD ROM / USB. Not via qemu or Virual Box. But the first thing you need to check is to run it on qemu, then on Virual Box.
```
cmd.exe > qemu-system-x86_64 -drive format=raw,file=boot_qemu_x86.bin
```

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/109%20-%2028-09-2024%20-%20qemu%20test%20bootloader.png?raw=true)

<hr>
Each .asm files were compiled by this command 

```
>nasm -f bin -o demo3.bin demo3.asm
>nasm -f bin -o bootloader.bin bootloader.asm
>nasm -f bin -o boot_qemu_x86.bin boot_qemu_x86.asm
```
NASM version
```
C:\Users\kdhome>nasm -v
NASM version 2.16.03 compiled on Apr 17 2024
```
<hr>
Example step by step <br />
1. Bootloader Code (Assembly) <br />

```
; bootloader.asm - A simple bootloader that prints "Hello World!" on the screen
; and hangs. It should be compiled with NASM and can be loaded using an emulator
; or burned to a bootable medium (like USB).

BITS 16                ; We are working in 16-bit real mode

start:
    ; Clear the screen by setting the video memory to all spaces (0x20)
    mov ax, 0xB800      ; Video memory segment (text mode)
    mov es, ax          ; ES = Video memory segment
    xor di, di          ; DI = 0, so we write to the start of video memory
    mov cx, 2000        ; Clear the first 2000 bytes (80x25 screen)

clear_loop:
    mov al, 0x20        ; ' ' (space character)
    mov ah, 0x07        ; White text on black background
    stosw               ; Write AL and AH (character and attribute) to memory
    loop clear_loop     ; Repeat until screen is clear

    ; Print "Hello World!" on the screen at the top left (position 0)
    mov si, msg         ; SI = address of the message
    mov di, 0           ; DI = 0 (start of video memory)
    
    call print_string   ; Print the string

hang:
    jmp hang            ; Infinite loop to prevent bootloader from continuing

print_string:
    ; Print a null-terminated string
    mov ah, 0x0E        ; BIOS teletype function for text output (int 10h)
print_loop:
    lodsb               ; Load byte from string (SI) to AL
    cmp al, 0           ; Is it the null terminator?
    je done             ; If yes, we're done printing
    int 0x10            ; Otherwise, print character in AL
    jmp print_loop      ; Repeat for next character
done:
    ret                 ; Return to caller

msg db "Hello World!", 0  ; Null-terminated string

times 510-($-$$) db 0     ; Pad with zeros until byte 510
dw 0xAA55                 ; Boot signature
```

2. Explanation of the Code
Clear Screen: The code clears the screen by writing spaces to video memory.
Print "Hello World!": The message "Hello World!" is printed on the screen using BIOS interrupt int 0x10.
Halt: After printing, the program enters an infinite loop, effectively halting the system.
The bootloader ends with a boot signature (0xAA55) at the last two bytes, required by the BIOS to recognize the bootloader as valid.
<br /><br />
3. Compilation Instructions <br />
Install NASM.  On Windows:
Download the NASM binary from the official website: https://www.nasm.us/
Follow the installation instructions provided there.
<br />
On linux :

```
sudo apt update
sudo apt install nasm
```
4. Compile the Bootloader

```
nasm -f bin -o bootloader.bin bootloader.asm
```

5. Running the Bootloader
<br /><br />
You can run the bootloader on real hardware (like a USB drive) or in a virtual machine (VM). Below are instructions for both methods.

Running on a USB Drive (Real Hardware)
Prepare a USB Drive (Make sure you have a backup of your data as this will overwrite the USB content).

On Linux, use dd to write the bootloader to the USB drive:

```
sudo dd if=bootloader.bin of=/dev/sdX bs=512 count=1
```
Replace /dev/sdX with your actual USB device (use lsblk to check).

On Windows, you can use a tool like Rufus to write the binary to a USB drive.

Boot from the USB Drive:

Restart your computer, enter the BIOS/UEFI settings, and select the USB drive as the boot device.
The system should boot from the USB drive and display "Hello World!" on the screen.
Running in a Virtual Machine (VM)
You can easily test your bootloader using an emulator like QEMU or VirtualBox.
Running with QEMU
Install QEMU:

On Linux: sudo apt install qemu
On Windows: Download from QEMU official website.
Run the bootloader:

```
qemu-system-x86_64 -drive format=raw,file=bootloader.bin
```

This will launch QEMU and you should see the "Hello World!" message printed on the screen.

Running with VirtualBox
Create a Virtual Hard Disk:

Use VBoxManage (comes with VirtualBox) to create a VHD from the bootloader binary:

```
VBoxManage convertfromraw bootloader.bin bootloader.vhd --format VHD
```
Create a Virtual Machine:

Open VirtualBox and create a new VM.
Choose Other for the OS type and use the VHD file as the virtual hard disk.
Start the VM, and the "Hello World!" message should appear.
<br /><br />
6. Demo alongside Windows / Linux<br /><br />
If you want to run this bootloader alongside an existing Windows or Linux installation, you can dual-boot it from the USB drive or run it in a VM as described above without affecting your existing OS installation.

To dual-boot:

Install the bootloader on a separate bootable USB drive.
Enter the BIOS/UEFI and configure the boot order to boot from the USB when it's connected, allowing you to switch between your regular OS and the bootloader demo.
<br /><br />
7. Write UEFI Bootloader (Assembly + C) - First, a bootloader targeting UEFI needs to be written in C or assembly. UEFI doesn't use 16-bit real mode like legacy BIOS; instead, it works in a more modern environment (protected or long mode). Here's a basic example of a UEFI bootloader that prints "Hello World!" on the screen in C.

```
// uefi_bootloader.c
#include <efi.h>
#include <efilib.h>

EFI_STATUS
EFIAPI
efi_main(EFI_HANDLE ImageHandle, EFI_SYSTEM_TABLE *SystemTable) {
    // Initialize UEFI Application
    InitializeLib(ImageHandle, SystemTable);

    // Print Hello World!
    Print(L"Hello World!\n");

    // Wait for a key press before exiting
    SystemTable->ConIn->Reset(SystemTable->ConIn, FALSE);
    EFI_INPUT_KEY Key;
    while ((SystemTable->ConIn->ReadKeyStroke(SystemTable->ConIn, &Key)) == EFI_NOT_READY);

    return EFI_SUCCESS;
}
```
8.Compile the UEFI Bootloader

To compile the UEFI bootloader, you need an environment that supports compiling UEFI applications. The most common toolchain is the GNU-EFI for Linux or Visual Studio with UDK for Windows.

On Linux (using GNU-EFI):
```
sudo apt-get install gnu-efi
gcc -I /usr/include/efi -I /usr/include/efi/x86_64 -nostdlib -fno-stack-protector -fpic -fshort-wchar -mno-red-zone -c uefi_bootloader.c -o uefi_bootloader.o
ld -nostdlib -znocombreloc -T /usr/lib/elf_x86_64_efi.lds -shared -Bsymbolic -L /usr/lib -L /usr/lib64 /usr/lib/crt0-efi-x86_64.o uefi_bootloader.o -o uefi_bootloader.so -lefi -lgnuefi
objcopy -j .text -j .sdata -j .data -j .dynamic -j .dynsym -j .rel -j .rela -j .reloc --target=efi-app-x86_64 uefi_bootloader.so uefi_bootloader.efi
```
On Windows (using Visual Studio and UDK):
Set up the EDK2 environment.
Compile your UEFI application through the Visual Studio toolchain.
<br /><br />
9.  Create a UEFI Bootable ISO - A UEFI-compatible ISO needs a specific file structure, including an EFI/BOOT/BOOTx64.EFI file on the disk. Here's how you can create this structure and pack it into an ISO image using Python.

Python Script to Create UEFI ISO
```
import os
import shutil
import subprocess

# Paths
efi_dir = "uefi_iso/EFI/BOOT"
iso_name = "bootable_uefi.iso"
bootloader_efi = "uefi_bootloader.efi"

# Create directories
os.makedirs(efi_dir, exist_ok=True)

# Copy the UEFI bootloader to the correct location
shutil.copy(bootloader_efi, os.path.join(efi_dir, "BOOTX64.EFI"))

# Use `genisoimage` or `mkisofs` to create a UEFI bootable ISO
# On Linux, `genisoimage` or `mkisofs` can be used to generate ISO images.
# On Windows, you may use a tool like `oscdimg`.
subprocess.run([
    "genisoimage", "-o", iso_name,
    "-b", "EFI/BOOT/BOOTX64.EFI",  # El Torito boot sector
    "-no-emul-boot", "-boot-load-size", "4", "-boot-info-table",
    "-eltorito-alt-boot", "-eltorito-platform", "efi", "-eltorito-boot", "EFI/BOOT/BOOTX64.EFI",
    "-no-emul-boot", "uefi_iso"
], check=True)

print(f"ISO created: {iso_name}")
```
This script assumes that:

You have genisoimage or mkisofs installed (for Linux).
The uefi_bootloader.efi file is already compiled and placed in the same directory as the Python script.
If you're on Windows, you can use the oscdimg tool from the Windows ADK to create the ISO:
```
oscdimg -u2 -udfver102 -bootdata:2#p0,e,bEFI\BOOT\BOOTx64.EFI -o uefi_iso bootable_uefi.iso
```
Step 4: Burn the ISO to a DVD/CD
Option 1: Burn the ISO using Python
You can use the os or subprocess modules in Python to call external DVD burning tools like cdrtools (for Linux) or IMGBurn (for Windows).

Here’s an example using the subprocess module to invoke wodim (Linux) or IMGBurn (Windows).

On Linux (using wodim):
```
import subprocess

# Path to your ISO file and DVD device
iso_path = "bootable_uefi.iso"
dvd_device = "/dev/sr0"  # Replace with your DVD device

# Burn the ISO to a DVD
subprocess.run([
    "wodim", "-v", "-dev=" + dvd_device, "-dao", iso_path
], check=True)

print("Burning completed!")
```
On Windows (using IMGBurn):
```
import subprocess

# Path to your ISO file and DVD drive
iso_path = "C:\\path\\to\\bootable_uefi.iso"
dvd_device = "D:"  # Replace with your DVD drive

# Burn the ISO to a DVD using IMGBurn
subprocess.run([
    "C:\\path\\to\\IMGBurn.exe", "/MODE", "WRITE", "/SRC", iso_path, "/DEST", dvd_device, "/START", "/CLOSE"
], check=True)

print("Burning completed!")
```
Option 2: Burn the ISO using C and WinAPI (Windows)
Here’s how you can do it in C on Windows using the WinAPI. Windows doesn’t have native support for burning discs through WinAPI, so typically, you would rely on external libraries like IMAPI2 for disc burning.

A simplified example that would require setting up COM objects to burn the disc can be quite lengthy and is better handled by utilities like IMGBurn or via Python as shown earlier.
<br /><br />
10. Booting from the CD/DVD - Once you’ve burned the ISO to a DVD/CD, you can boot from it on a UEFI-compatible system:

Insert the DVD into your laptop’s DVD drive.
Reboot the laptop and press the appropriate key (usually F12, Esc, Del, or F2) to access the boot menu.
Select the DVD drive as the boot option.
If everything is set up correctly, the system should boot and display "Hello World!" on the screen.

https://f.osdev.org/viewtopic.php?t=28894

<br /><br />
To convert a bootloader.bin file into an .iso format in Python or C/WinAPI, we need to ensure the binary bootloader (bootloader.bin) is wrapped in a proper bootable ISO structure. Below, I'll show how you can do this both using Python and C/WinAPI.

1. Using Python:
In Python, we can use an external tool like mkisofs or genisoimage to create a bootable ISO. These tools are readily available on Linux systems, but they can also be installed on Windows using Cygwin or WSL (Windows Subsystem for Linux).

Here is a Python script to create an ISO image from a bootloader binary:

Python Script to Create Bootable ISO
```
import os
import subprocess

# Paths
bootloader_bin = "bootloader.bin"   # Path to the bootloader binary
iso_name = "bootable_iso.iso"       # Name of the output ISO
iso_dir = "iso_dir"                 # Temporary directory for ISO contents

# Create the directory structure for the ISO
os.makedirs(iso_dir, exist_ok=True)

# Copy the bootloader to the root of the ISO directory
bootloader_path = os.path.join(iso_dir, "bootloader.bin")
with open(bootloader_bin, 'rb') as f_src, open(bootloader_path, 'wb') as f_dst:
    f_dst.write(f_src.read())

# Use mkisofs or genisoimage to create a bootable ISO
# Ensure you have mkisofs or genisoimage installed. On Linux, install with:
# sudo apt-get install genisoimage

subprocess.run([
    "genisoimage", "-o", iso_name,  # Output ISO file
    "-b", "bootloader.bin",         # The bootable binary file
    "-no-emul-boot",                # Treat the boot image as a raw binary
    "-boot-load-size", "4",         # Number of 512-byte sectors to load
    "-boot-info-table",             # Add a boot info table in the boot image
    iso_dir                         # The directory containing ISO contents
], check=True)

print(f"ISO created: {iso_name}")
```
How It Works:
Directory Setup:
We create a temporary directory (iso_dir) where we place the bootloader.bin file.
Create ISO:
We use genisoimage or mkisofs to generate the bootable ISO. The -b option tells the tool that bootloader.bin is the bootable file, and -no-emul-boot ensures it's treated as a raw binary.
Run the Script:
After running this Python script, it will produce a bootable ISO file.
2. Using C and WinAPI:
On Windows, creating a bootable ISO file programmatically using WinAPI is not straightforward, as there's no native API for this. However, we can invoke external tools from C, such as oscdimg.exe from the Windows Assessment and Deployment Kit (ADK), or use a library that can handle ISO creation, like libburnia.

Here’s how you can write a simple C program that invokes oscdimg to convert bootloader.bin into a bootable ISO:

Step-by-Step Guide Using oscdimg.exe
Install the Windows ADK:

Download and install the Windows ADK to get the oscdimg.exe tool.
C Code to Call oscdimg.exe:

Here’s a simple C program that uses oscdimg.exe to create an ISO:
```
#include <windows.h>
#include <stdio.h>

int main() {
    // Paths
    const char* oscdimg_path = "C:\\Program Files (x86)\\Windows Kits\\10\\Assessment and Deployment Kit\\Deployment Tools\\Oscdimg\\oscdimg.exe";
    const char* bootloader_bin = "C:\\path\\to\\bootloader.bin";
    const char* iso_name = "C:\\path\\to\\output.iso";
    const char* iso_dir = "C:\\path\\to\\iso_dir";  // Directory where bootloader is placed

    // Command to run oscdimg to create an ISO
    char command[1024];
    snprintf(command, sizeof(command), "\"%s\" -b%s -o %s %s", oscdimg_path, bootloader_bin, iso_name, iso_dir);

    // Run the command
    printf("Running command: %s\n", command);
    int result = system(command);

    // Check if oscdimg executed successfully
    if (result == 0) {
        printf("ISO successfully created: %s\n", iso_name);
    } else {
        printf("Failed to create ISO.\n");
    }

    return result;
}
```
How It Works:
Oscdimg Usage:

oscdimg is a command-line tool provided by Microsoft, capable of generating ISO images, including bootable ones.
Command Construction:

The program constructs a command to invoke oscdimg.exe, passing it the paths to the bootloader binary (-b) and the directory containing the bootloader (iso_dir). The output ISO is specified (-o).
Running the Command:

The system() call runs the command to create the ISO.
3. Burning the ISO to a CD/DVD
Once you have the .iso file (created by either method), you can use any burning software (e.g., IMGBurn, Brasero, or wodim) to burn the ISO to a CD/DVD. On Windows, you could automate this burning process using IMAPI or external tools like IMGBurn.

For Python, the previous example shows how to call wodim or IMGBurn to burn the ISO to a CD/DVD.

4. Building ISO from Scratch in C (Low-Level):
Creating a raw ISO file from scratch in C without external tools is quite complex. You'd need to:

Create the appropriate directory structures.
Write the El Torito boot catalog and boot sectors.
Write the file system metadata (ISO 9660, UDF, or Joliet) for the ISO.
This involves adhering to the ISO 9660 file system standard and manually creating the necessary file and directory structures. Typically, libraries like libburn or libisofs are used to avoid reinventing the wheel.

Final Thoughts:
Using oscdimg.exe or tools like genisoimage in Python is the easiest and most reliable approach to convert a bootloader.bin into a bootable ISO file. If you’re specifically on Windows and want to avoid external tools, you’d need to delve into complex low-level ISO creation code or use a third-party library like libburn or libisofs.
<hr>
The VBoxManage convertfromraw command in VirtualBox is specifically designed to convert a raw disk image (such as bootloader.bin) into a virtual hard disk (VHD, VDI, or VMDK) format. It cannot be used directly to create ISO files, which are fundamentally different from disk images or virtual hard drives.

An ISO file is a disk image designed to represent the contents of an optical disc (CD/DVD), typically with a filesystem like ISO 9660 or UDF, while a VHD is a virtual hard drive format, meant to simulate a physical hard disk.

That said, VirtualBox's VBoxManage utility cannot create ISO files. However, there are alternative methods to create an ISO file from a binary bootloader using tools like mkisofs or genisoimage, as I described earlier.

Overview of the Differences:
bootloader.vhd: This would be a virtual hard disk file, which you could use in VirtualBox as a virtual hard drive, but not as a bootable ISO.
bootloader.iso: This is a CD/DVD image, bootable and used for optical disc emulation. This is the format you want for booting in environments expecting an ISO.
What Can You Do Instead?
If you're specifically aiming to create an ISO file from your bootloader binary, VBoxManage is not the tool you should use. Here's how you can achieve this instead:

Create a bootable ISO using mkisofs/genisoimage:
You can use these utilities to create an ISO from the bootloader.bin file.

Command Example:
```
genisoimage -o bootloader.iso -b bootloader.bin -no-emul-boot -boot-load-size 4 -boot-info-table iso_dir
```
Explanation:
-o bootloader.iso: The output ISO file.
-b bootloader.bin: Specifies that bootloader.bin should be the bootable image.
-no-emul-boot: Treats the binary file as a raw bootloader image (required for El Torito bootable CDs).
iso_dir: The directory containing bootloader.bin or other files you want in the ISO.
Use VBoxManage in combination with the ISO file:
Once the ISO is created, you can attach it to a VirtualBox VM. VirtualBox can boot from ISO files directly:
```
VBoxManage storageattach <vmname> --storagectl "IDE Controller" --port 0 --device 0 --type dvddrive --medium bootloader.iso
```
Summary:
No, VBoxManage cannot create an ISO file, but it can convert raw binary files to virtual hard disk formats (VHD/VDI/VMDK).
To create a bootable ISO, use tools like mkisofs or genisoimage.
<br /><br />
To clarify:

VirtualBox can boot from ISO files but not from raw binaries (bootloader.bin) or VHD files as a CD/DVD drive.
You need to convert your raw binary bootloader (bootloader.bin) to an ISO file if you want to boot it as a virtual CD/DVD in VirtualBox.
Step-by-Step: Convert bootloader.bin to bootloader.iso and Load in VirtualBox
To make your bootloader work in VirtualBox, you can create an ISO image from bootloader.bin using a tool like genisoimage or mkisofs. After that, you can load the ISO file into VirtualBox.

Here's how to do it:

Step 1: Convert bootloader.bin to ISO
If you're on Linux, macOS, or using WSL (Windows Subsystem for Linux), you can use genisoimage (or mkisofs).

Command to Create an ISO

```
genisoimage -o bootloader.iso -b bootloader.bin -no-emul-boot -boot-load-size 4 -boot-info-table ./iso_root
```

```
-o bootloader.iso: This specifies the output ISO file (bootloader.iso).
-b bootloader.bin: This tells the tool that bootloader.bin should be the boot image.
-no-emul-boot: Tells the tool that the image is a raw binary (this is needed for bootloaders like your .bin file).
-boot-load-size 4: Specifies the number of 512-byte sectors to load (adjust if needed).
-boot-info-table: Adds a boot info table for compatibility.
./iso_root: A directory that holds the files for the ISO. You can include any files you need in this folder. The bootloader.bin must also be copied here.
If you don't need any additional files, you can keep ./iso_root as an empty directory.
```

Example:
Assume the following:

You have a bootloader.bin file.
You want to create a bootable ISO.
```
mkdir iso_root
cp bootloader.bin iso_root/
genisoimage -o bootloader.iso -b bootloader.bin -no-emul-boot -boot-load-size 4 -boot-info-table iso_root
```
This will generate a bootloader.iso file that you can use in VirtualBox.

On Windows:
On Windows, you can use Cygwin to install genisoimage or use third-party tools like MagicISO or UltraISO to manually create an ISO from your bootloader.

Alternatively, you can use oscdimg.exe (part of the Windows ADK) with the following command:
```
oscdimg -bbootloader.bin -no-emul-boot -boot-load-size 4 -boot-info-table iso_root bootloader.iso
```
Step 2: Load the ISO into VirtualBox
Open VirtualBox.
Select your VM (or create a new VM if needed).
Go to the settings of your VM.
Navigate to the "Storage" section.
Click on the empty optical drive under the "Controller: IDE" or "Controller: SATA".
Click the small CD icon next to the "Optical Drive" dropdown.
Choose a disk file and select your newly created bootloader.iso.
Start the VM.
Additional Notes:
Make sure your bootloader is correctly written to be loaded by BIOS/UEFI. If your bootloader is too simple (e.g., it only works in real mode and expects to be loaded at the beginning of the disk), you may need to write some BIOS-compatible boot code.
If your bootloader is UEFI-based, ensure that you're using the UEFI firmware in VirtualBox. You can enable UEFI mode by going to Settings → System → Enable EFI.
Example Flow:
```
# 1. Create directory for ISO contents
mkdir iso_root

# 2. Copy bootloader into directory
cp bootloader.bin iso_root/

# 3. Create ISO file from bootloader
genisoimage -o bootloader.iso -b bootloader.bin -no-emul-boot -boot-load-size 4 -boot-info-table iso_root/

# 4. Load bootloader.iso in VirtualBox and boot the VM!
```
Conclusion:
You cannot use VBoxManage convertfromraw to create an ISO—it's for virtual disk images, not ISOs.
Use tools like genisoimage or mkisofs to convert your bootloader.bin into an ISO file.
Once you have the ISO, you can load it into VirtualBox as a bootable CD/DVD.
This approach will let you boot your bootloader.bin file in VirtualBox! 
<hr>
1. Bootloader Issues
Before diving into the ISO creation itself, make sure that the bootloader is functioning properly. If the bootloader is not correctly handling the boot process, even a perfectly generated ISO won't work in VirtualBox.

Ensure your bootloader is written to be bootable and that it adheres to the x86 boot process (i.e., it starts at the boot sector and loads at address 0x7C00).
If this bootloader has been tested directly with qemu or bochs and works, it’s likely that the issue is related to the ISO creation or VirtualBox settings.
2. Creating a Minimal Bootloader ISO
To confirm whether the issue is with the ISO creation process, let's first focus on creating a minimal bootable ISO that should work in VirtualBox.

Minimal Bootloader Example:
Below is a simple 16-bit assembly bootloader, which prints "Hello, World!" and then hangs:
```
[BITS 16]
[ORG 0x7C00]

start:
    mov si, msg              ; Point to the message
    call print_string         ; Print the message

hang:
    cli                      ; Disable interrupts
    hlt                      ; Halt the CPU (endless loop)
    
print_string:
    mov ah, 0x0E              ; Teletype output function (BIOS service)
.repeat:
    lodsb                    ; Load next byte of the string
    test al, al               ; Check if the byte is zero (end of string)
    jz .done                 ; If zero, we're done
    int 0x10                 ; Print character
    jmp .repeat              ; Repeat for next character
.done:
    ret                      ; Return to caller

msg db "Hello, World!", 0

times 510-($-$$) db 0         ; Pad with zeros to make 512 bytes (boot sector size)
dw 0xAA55                    ; Boot sector signature
```
Steps to Assemble and Create an ISO (Linux):
Assemble the Bootloader:
```
nasm -f bin -o bootloader.bin bootloader.asm
```
Prepare the ISO Directory:
Create a directory where the bootloader will reside. This directory will act as the root of the ISO image.
```
mkdir iso_dir
cp bootloader.bin iso_dir/
```
Create the ISO with genisoimage:
Use genisoimage to create the ISO. Make sure to pass the correct options for booting:
```
genisoimage -o bootloader.iso -b bootloader.bin -no-emul-boot -boot-load-size 4 -boot-info-table iso_dir/
```
-b bootloader.bin: This specifies that bootloader.bin is the bootable image.
-no-emul-boot: Tells the tool to treat the boot image as a raw binary (no floppy or hard disk emulation).
-boot-load-size 4: Ensures that 4 sectors (512 bytes each) are loaded.
-boot-info-table: Adds information to the boot image, useful for certain boot configurations.
Test with qemu:
To quickly test if the ISO works before using VirtualBox, you can use qemu to boot the ISO:
```
qemu-system-x86_64 -cdrom bootloader.iso
```
If the ISO works correctly in qemu, you should see "Hello, World!" printed.

3. Load the ISO in VirtualBox
Once the ISO is confirmed to be working, you can proceed with VirtualBox:

Open VirtualBox and create a new VM:

Choose the Operating System type as "Other/Unknown" and set the base memory to at least 32MB.
Load the ISO file:

Go to the Storage settings of the VM.
Select Empty under the "Controller: IDE" section.
Click the CD icon next to "Optical Drive" and choose "Choose a disk file...".
Select the bootloader.iso you created.
Boot the VM:

Start the VM, and if everything is set up correctly, it should boot from the ISO and execute the bootloader.
4. Troubleshooting:
If the VM doesn't boot or you encounter issues, consider the following:

Check the VirtualBox Logs:

Go to the VM settings → Logs and review the boot process logs for any errors.
Make Sure to Enable BIOS Boot (Not EFI):

VirtualBox supports both BIOS and EFI booting. For a simple bootloader like this, BIOS booting is needed.
Disable EFI in the System settings of your VM.
Test Different Memory and CPU Settings:

Sometimes, VirtualBox behaves differently based on the memory allocation or CPU settings. Try adjusting these in the System section.
5. Check Bootloader File Layout:
Make sure your bootloader is structured correctly:

The boot sector must be 512 bytes in size.
The last two bytes of the boot sector must contain the boot signature 0xAA55. This is essential for the BIOS to recognize it as a bootable disk.
If you're not seeing anything happen, it's possible the bootloader.bin is either not correctly formatted as a boot sector, or VirtualBox is not reading the ISO properly.

Conclusion:
Ensure the bootloader is properly assembled and formatted for BIOS booting.
Use genisoimage with the correct options to create the ISO (-no-emul-boot for raw binaries).
Test the ISO in qemu to make sure it boots before trying VirtualBox.
Load the ISO in VirtualBox, ensuring BIOS booting (not UEFI) is enabled.
<hr>
Common Issues to Check
Verify the ISO Structure:

After running the genisoimage command, ensure that the bootloader.iso is created and is approximately 1.4MB or smaller.

Check if the generated ISO is correctly structured. You can use tools like isoinfo (part of genisoimage) to check the content:
```
isoinfo -l -i bootloader.iso
```
This command should list bootloader.bin as part of the contents of the ISO.

Check the Bootloader Signature:

Make sure the bootloader binary ends with the bytes 0x55 and 0xAA. This is crucial for the BIOS to recognize it as a bootable image. You can inspect the last two bytes using a hex editor or a command like:
```
xxd bootloader.bin | tail -n 1
```
The output should show the last line ending with 55 AA.

Adjust BIOS/UEFI Settings in QEMU:

If you are using a specific BIOS/UEFI option in QEMU, try switching it. Generally, you would want to use the default BIOS for a simple bootloader.
Ensure Permissions:

Ensure you have the necessary permissions to read the files and create the ISO.
Alternative Testing with QEMU
If you're still facing issues, you can also try a more verbose QEMU command for debugging:
```
qemu-system-x86_64 -cdrom bootloader.iso -boot d -m 64 -smp 1 -monitor stdio
```
Conclusion
Ensure the bootloader binary is properly structured with a correct boot sector.
Create the ISO carefully, using genisoimage with the right options.
Validate the ISO structure to ensure the bootloader is included correctly.
<hr>
Booting in QEMU
You can use the -drive option to specify the ISO file as a hard disk image:
```
qemu-system-x86_64 -drive format=raw,file=bootloader.iso,if=virtio -boot d
```
Explanation of Command Options:
-drive: This option is used to specify a disk image.
format=raw: Specifies that the image is a raw format.
file=bootloader.iso: Path to your ISO file.
if=virtio: This sets the interface type to VirtIO, which can offer better performance, though you can also omit this option to use the default IDE interface.
-boot d: This tells QEMU to attempt to boot from the first drive (in this case, the ISO).
Alternative with -hda
You can also use -hda if you prefer:
```
qemu-system-x86_64 -hda bootloader.iso -boot d
```
Summary
Create and assemble your bootloader.
Create a bootable ISO with the bootloader included.
Boot from the ISO using QEMU with the -drive or -hda option.
Troubleshooting
If you encounter the "No bootable device" error again:
Double-check that the ISO file is structured properly and includes the bootable bootloader.
Verify that the last two bytes of your bootloader binary are 0x55 and 0xAA for it to be recognized as bootable.
Conclusion
This method allows you to boot directly from an ISO file in QEMU as if it were a hard disk image, bypassing the need for CD-ROM emulation.
<hr>
REFERENCES: <br />
https://github.com/XaHertz/Windows-8.1-Pro-WMC-ISO-Tool<br />
https://learn.microsoft.com/en-us/windows-hardware/get-started/adk-install#download-the-adk-101253981-september-2023<br />
https://f.osdev.org/viewtopic.php?t=28894<br />
https://linux.die.net/man/8/isoinfo <br />
https://stackoverflow.com/questions/31831268/genisoimage-and-uefi <br />
https://www.tecmint.com/create-iso-from-directory-linux/ <br />
https://www.howtogeek.com/devops/how-to-use-qemu-to-boot-another-os/ <br />
https://gist.github.com/aalemayhu/0c8664e82c7f6e554e30aeecf74dffad <br />
https://linux-tips.com/t/booting-from-an-iso-image-using-qemu/136 <br />
https://github.com/agx-r/Bootloader/blob/main/bootloader.asm <br />
https://www.youtube.com/watch?v=xFrMXzKCXIc&ab_channel=NirLichtman
<hr>
Final thoughts. Back few years ago... I build on linux booloader and using this command

```
menuentry "myos"{
	multiboot /boot/myos.bin
}

```
https://github.com/pritamzope/OS/tree/master/Kernel/Simple/src/kernel_1 <br />
https://wiki.osdev.org/Printing_To_Screen <br />
<br /><br />
main.c
```
// note this example will always write to the top
// line of the screen
void main( int colour, const char *string )
{
    volatile char *video = (volatile char*)0xB8000;
    while( *string != 0 )
    {
        *video++ = *string++;
        *video++ = colour;
    }
}
```

linker.ld

```
/* entry point of our kernel */
ENTRY(_start)

SECTIONS
{
	/* we need 1MB of space atleast */
	. = 1M;

  	/* text section */
	.text BLOCK(4K) : ALIGN(4K)
	{
		*(.multiboot)
		*(.text)
	}

	/* read only data section */
	.rodata BLOCK(4K) : ALIGN(4K)
	{
		*(.rodata)
	}

	/* data section */
	.data BLOCK(4K) : ALIGN(4K)
	{
		*(.data)
	}

	/* bss section */
	.bss BLOCK(4K) : ALIGN(4K)
	{
		*(COMMON)
		*(.bss)
	}

}
```

boot.s

```
# set magic number to 0x1BADB002 to identified by bootloader 
.set MAGIC,    0x1BADB002

# set flags to 0
.set FLAGS,    0

# set the checksum
.set CHECKSUM, -(MAGIC + FLAGS)

# set multiboot enabled
.section .multiboot

# define type to long for each data defined as above
.long MAGIC
.long FLAGS
.long CHECKSUM


# set the stack bottom 
stackBottom:

# define the maximum size of stack to 512 bytes
.skip 1024


# set the stack top which grows from higher to lower
stackTop:

.section .text
.global _start
.type _start, @function


_start:

  # assign current stack pointer location to stackTop
	mov $stackTop, %esp

  # call the kernel main source
	call main

	cli


# put system in infinite loop
hltLoop:

	hlt
	jmp hltLoop

.size _start, . - _start
```

run.sh

```
#assemble boot.s file
as --32 boot.s -o boot.o

#compile kernel.c file
gcc -m32 -c main.c -o main.o -std=gnu99 -ffreestanding -O2 -Wall -Wextra

#linking the kernel with kernel.o and boot.o files
ld -melf_i386 -T linker.ld main.o boot.o -o mytestos.bin -nostdlib
```

This produce .bin file. And then <br />
grub.cfg

```
menuentry "MyOS" {
	multiboot /boot/MyOS.bin
}
```

run.sh

```
#assemble boot.s file
as --32 boot.s -o boot.o

#compile kernel.c file
gcc -m32 -c kernel.c -o kernel.o -std=gnu99 -ffreestanding -O2 -Wall -Wextra

#linking the kernel with kernel.o and boot.o files
ld -m elf_i386 -T linker.ld kernel.o boot.o -o MyOS.bin -nostdlib

#check MyOS.bin file is x86 multiboot file or not
grub-file --is-x86-multiboot MyOS.bin

#building the iso file
mkdir -p isodir/boot/grub
cp MyOS.bin isodir/boot/MyOS.bin
cp grub.cfg isodir/boot/grub/grub.cfg
grub-mkrescue -o MyOS.iso isodir

#run it in qemu
qemu-system-x86_64 -cdrom MyOS.iso
```

But this create .iso file int linux system. And using gtub craete on the booting list menu entry point to this kernel. This is not load kernel from CD/DVD. but it worked from what I remember.
