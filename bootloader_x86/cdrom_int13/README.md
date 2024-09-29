[PL]
Są tutaj 2 wersje 10 oraz 11. 10 ma prawdopodobnie problem tego typu
```
The error message logs you provided from QEMU indicate that the bootloader is experiencing issues that are likely related to how it interacts with the emulated hardware, particularly around the hardware interrupt handling (INT 0x08) and possibly the Segment Descriptor Table (GDT) and Interrupt Descriptor Table (IDT).

The relevant logs show a continuous servicing of hardware interrupt 0x08, which is typically related to the CPU timer. The continuous check_exception logs suggest that there are unhandled exceptions being raised during execution. Below are several steps and recommendations to address these issues and refine the bootloader code.

Possible Issues and Solutions
Properly Set Up Segment Registers: Make sure the segment registers are initialized correctly. In real mode, segment registers must be set up to point to the right memory addresses. The default segment values should be set for ES, CS, DS, SS, and so on.

Ensure Correct Use of the Stack: You should set up the stack pointer (ESP) properly before using stack operations. This is particularly crucial in 16-bit mode where you have to manage the stack carefully.

Check for Infinite Loops: The handling of interrupts should be confirmed to ensure that they are not stuck in an infinite loop. If you're generating too many interrupts without servicing them properly, it could lead to such behavior.
```
W wersji 11 jest to poprawione na początku i przy uruchomieniu z komendy  
```
qemu-system-x86_64 -cdrom boot10.iso -boot d -m 64 -d int
```
pokazuje coś takiego na końcu i się nie zatrzymuje. Jakiś błąd.
```
check_exception old: 0xffffffff new 0x6
check_exception old: 0xffffffff new 0x6
check_exception old: 0xffffffff new 0x6
Servicing hardware INT=0x08
check_exception old: 0xffffffff new 0x6
check_exception old: 0xffffffff new 0x6
check_exception old: 0xffffffff new 0x6
check_exception old: 0xffffffff new 0x6
check_exception old: 0xffffffff new 0x6
check_exception old: 0xffffffff new 0x6
```
<hr>
Ale Wersja 11 działa już ok. To jest komenda która uruchamia .BIN z -drive i to działa co widać na obrazku

```
qemu-system-x86_64 -drive format=raw,file=boot11.bin -d int
```

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/cdrom_int13/113%20-%2029-09-2024%20-%20to%20dziala.png?raw=true)

Następnie -cdrom. I tutaj z tego co widzę jest kilka tematów związanych z IN 13 ale nie mam pewności <br />
https://board.flatassembler.net/topic.php?t=13607 <br />
https://wiki.osdev.org/Bootloader <br />
https://github.com/mahdi2019/simple-bootloader/tree/master <br />
https://stackoverflow.com/questions/25454487/i-cant-read-a-sector-from-a-cd-rom-in-x86-assembly <br />
https://en.wikibooks.org/wiki/X86_Assembly/Bootloaders <br />
https://forum.osdev.org/viewtopic.php?t=20720 <br />
https://git.kernel.org/pub/scm/boot/syslinux/syslinux.git/tree/core/isolinux.asm?id=HEAD <br />
https://medium.com/@g33konaut/writing-an-x86-hello-world-boot-loader-with-assembly-3e4c5bdd96cf <br />
https://github.com/gotoco/PE_Bootloader_x86 <br />
https://en.wikipedia.org/wiki/INT_13H <br />

```
>qemu-system-x86_64 -cdrom boot11.iso -boot d -m 64 -d int
```

To jest poprawiona wersja 11. Dlatego nie ma błędów ```check_exception old: 0xffffffff new 0x6```
Ale zatrzymuje się tutaj. I nie ma nic poza tym. Ale to oznacza że bootuje tylko nie robi nic konkretnego dalej, albo się zawiesza na jakiejś instrukcji.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/cdrom_int13/112%20-%2029-09-2024%20-%20cd.png?raw=true)

<hr>
<h2>make a little progress</h2>
first of at all, I need some information about virtual hardware. About matherboard specificatin, CDROM etc. And this information on the basic side you find here. <br/>
https://www.qemu.org/docs/master/system/index.html <br />
https://www.qemu.org/docs/master/system/invocation.html#hxtool-8 <br />
https://www.qemu.org/docs/master/system/target-i386.html#board-specific-documentation <br />
https://www.qemu.org/docs/master/system/target-i386.html <br />
https://www.qemu.org/docs/master/system/target-i386.html#board-specific-documentation <br />
https://www.qemu.org/docs/master/system/i386/microvm.html <br />
https://www.qemu.org/docs/master/system/i386/pc.html <br />
<br />
But this is more deeply topic, about structure ISO disk etc https://en.wikipedia.org/wiki/Disk_sector
<br />
So, as you see... this is a lot of study !!! To run demos on Virtual Box you have to got study another docs :)) <br />
https://www.virtualbox.org/wiki/End-user_documentation<br />
https://www.virtualbox.org/manual/ch01.html#hostossupport<br />
<br />
But you need go deeper inside implementation... and again study study... etc https://github.com/qemu/qemu || https://github.com/qemu/qemu/tree/master/hw || https://github.com/qemu/qemu/tree/master/pc-bios
<br /><br />
<b>OK, back to QEMU !</b><br />
1. QEMU Command-Line Options for Debugging and System Information <br />
-machine Option: Display information about the emulated machine (e.g., chipset, board):
  
```
qemu-system-x86_64 -machine help
```

-cpu Option: Specify and get detailed information about the CPU being emulated. You can use qemu-system-x86_64 -cpu help to list available CPU models: (You can also use -cpu host to use the host CPU's features inside the VM.)<br />

```
qemu-system-x86_64 -cpu help
```

-smbios Option: This allows you to inspect or specify the BIOS System Management BIOS (SMBIOS) tables, which can give you information about the virtual machine’s hardware and firmware:

```
qemu-system-x86_64 -smbios type=0,vendor="CustomVendor",version="1.0"
```

-dump-vmstate Option: This is used to dump the VM’s state information (may include CPU, memory, and device state) to a file for debugging:

```
qemu-system-x86_64 -dump-vmstate dump-file-name
```

-device help: To get a list of available devices that can be emulated by QEMU (these include various hardware peripherals like network cards, sound devices, etc.):

```
qemu-system-x86_64 -device help
```

2. BIOS Information Using SeaBIOS/OVMF
When using qemu-system-x86_64, the BIOS or UEFI firmware can be either SeaBIOS (for legacy boot) or OVMF (for UEFI boot). To get system and BIOS information, you can do the following:

a. Access SeaBIOS Information:
When QEMU boots up with SeaBIOS, press F12 or Esc during the boot process to access the boot menu. SeaBIOS typically displays information about the BIOS version and the memory setup.
b. Access OVMF (UEFI):
If QEMU is using OVMF (UEFI firmware), you can press Esc during boot to access the UEFI menu. From there, you can navigate to see system information such as the firmware version, memory map, and available boot options.
<br /><br />
3. Use dmidecode to Get SMBIOS Information (Inside VM)
If you have a Linux operating system running inside the QEMU virtual machine, you can use the dmidecode utility to query the system's SMBIOS tables. This provides detailed information about the BIOS, motherboard, CPU, and other hardware.

Install dmidecode (on a Linux guest):

```
sudo apt-get install dmidecode  # For Debian/Ubuntu-based systems
sudo yum install dmidecode      # For RHEL/CentOS-based systems
```

Run dmidecode to get system information:

```
sudo dmidecode
```

This will give you a detailed output with information about:

BIOS version and vendor
System manufacturer and product name
Motherboard details (baseboard information)
CPU details
Memory modules
Extract specific information:
BIOS Information:

```
sudo dmidecode -t bios
```

System Information:

```
sudo dmidecode -t system
```

Baseboard (Motherboard) Information:

```
sudo dmidecode -t baseboard
```

4. Query CPU Information
Inside Linux: You can run lscpu or view the /proc/cpuinfo file inside a Linux VM to get detailed information about the virtualized CPU:

```
lscpu
cat /proc/cpuinfo
```

5. Query Memory Information
Inside Linux: You can use free -h or cat /proc/meminfo inside the Linux guest OS to get detailed information about the virtual memory and physical memory available in the QEMU virtual machine:

```
free -h
cat /proc/meminfo
```

6. Use info Commands in QEMU Monitor
QEMU has an integrated monitor interface that allows you to inspect the state of the emulated system in real time. To access the QEMU monitor, press Ctrl+Alt+2 in the QEMU window. Once in the monitor, you can use the following commands:

info version: Get the QEMU version.
info cpus: Display information about the virtual CPUs.
info registers: View the state of CPU registers.
info mem: Show memory mappings.
info mtree: Show the memory layout of the guest.
To return to the VM interface, press Ctrl+Alt+1.

7. Documentation on QEMU and System Information
To get more detailed documentation about QEMU, system emulation, and various options, refer to the following resources:

QEMU Wiki: The QEMU wiki provides detailed information on running QEMU, setting up virtual devices, and the various hardware platforms it emulates.

QEMU Wiki
QEMU Documentation: Run the following command to read the built-in QEMU documentation:

```
man qemu-system-x86_64
```

SeaBIOS Documentation: If using SeaBIOS, you can check the SeaBIOS project documentation here:

SeaBIOS Documentation
OVMF (UEFI Firmware for QEMU): The OVMF project is part of the TianoCore UEFI firmware implementation and provides UEFI support for QEMU.

OVMF Documentation
Summary of Useful Commands and Tools:
qemu-system-x86_64 -machine help (list available machine types)
qemu-system-x86_64 -cpu help (list available CPU models)
qemu-system-x86_64 -device help (list devices)
dmidecode (inside Linux VM to get BIOS/system details)
lscpu (CPU details inside Linux VM)
info commands in QEMU monitor (Ctrl+Alt+2 for QEMU monitor)
By using the above methods and tools, you can gather detailed information about the system, board, and BIOS in qemu-system-x86_64.

<h2>QEMU Monitor</h2>
When qemu-system-x86_64 is running, you can retrieve detailed information about the devices and hardware being emulated using the QEMU Monitor and other methods. Below are the main approaches to gather information about devices in a running QEMU virtual machine.

1. QEMU Monitor
QEMU includes an interactive monitor that can be accessed during runtime to inspect and control various aspects of the virtual machine, including the devices being emulated.

How to Access the QEMU Monitor
You can switch to the QEMU monitor interface by pressing Ctrl+Alt+2 while the QEMU virtual machine is running. To switch back to the VM interface, press Ctrl+Alt+1.

Once in the monitor, you can use the following commands to get device information:

Useful Monitor Commands:
info qtree: Displays the device hierarchy (tree view), which includes emulated devices, buses, and hardware components attached to the virtual machine.

```
info qtree
```

Example output:

```
bus: pci.0
  type PCI
  dev: 00:01.0, id ""
    type: virtio-net-pci
    pci id: 1af4:1000
    class id: 02:00:00
  dev: 00:02.0, id ""
    type: virtio-blk-pci
    pci id: 1af4:1001
    class id: 01:00:00
```

This shows devices such as virtio-net (network) and virtio-blk (block device like a virtual hard disk) on the PCI bus.

info pci: Shows information about the PCI devices in the virtual machine. Each device is shown with its bus, device, and function numbers, vendor ID, device ID, and device class.

```
info pci
```

Example output:

```
Bus  0, device   1, function 0:
  Virtio network device
    IRQ 10.
Bus  0, device   2, function 0:
  Virtio block device
    IRQ 11.
```

info usb: Displays USB devices connected to the virtual machine.

```
info usb
```

info block: Displays block devices (disks, CD-ROMs, etc.) attached to the virtual machine, including information about image files, formats, and sizes.

```
info block
```

Example output:

```
ide0-hd0 (#block202): /path/to/disk.img (raw)
    Cache mode: writeback
```

info irq: Shows the interrupts (IRQs) used by devices.

```
info irq
```

info cpus: Provides information about the CPU(s) in the virtual machine.

```
info cpus
```

info network: Displays network device information (tap devices, NICs, bridges, etc.).

```
info network
```

info status: Gives information about the current VM status (running, paused, etc.).

```
info status
```

2. QEMU Command-Line Options for Logging and Debugging
You can also enable certain logging options when you launch qemu-system-x86_64 to get real-time information about the devices and system configuration.

Enable Device Logging: Use -d followed by the type of logging you'd like to capture. For example, to log guest hardware device events:

```
qemu-system-x86_64 -d guest_errors
```

Device Model Debugging: Use -device help to list the devices QEMU can emulate. This can be useful to see if a specific device is in use.

```
qemu-system-x86_64 -device help
```

Example output:

```
name "virtio-net-pci", bus PCI, desc "Virtio network device"
name "virtio-blk-pci", bus PCI, desc "Virtio block device"
```

Add -monitor stdio: When starting QEMU, if you include the -monitor stdio option, you can interact with the QEMU monitor directly through the terminal in which QEMU is running:

```
qemu-system-x86_64 -monitor stdio
```

This way, you can use all info commands directly without switching interfaces.

3. Inspect Devices from Inside the Guest OS
You can also inspect the devices from within the guest OS (assuming you're running a Linux or other operating system inside the VM). Here are common methods for gathering device information from within the guest.

a. lspci (Linux)
The lspci command shows information about PCI devices within the VM. Install pciutils package if it’s not installed:

```
sudo apt-get install pciutils   # For Debian/Ubuntu-based distros
sudo yum install pciutils       # For RHEL/CentOS-based distros
```

Then run:

```
lspci
```

This will list all PCI devices, including virtual hardware like network interfaces and block devices.

b. lsusb (Linux)
To display USB devices, use lsusb inside the guest OS:

```
lsusb
```

c. lsblk (Linux)
To list block devices such as disks and partitions:

```
lsblk
```

d. dmesg (Linux)
The dmesg command outputs the kernel ring buffer and shows hardware-related events, including device detection at boot time:

```
dmesg | grep -i "device"
```

e. /proc Filesystem (Linux)
You can also explore system information through the /proc filesystem:

/proc/cpuinfo: CPU information

```
cat /proc/cpuinfo
```

/proc/meminfo: Memory information.

```
cat /proc/meminfo
```

4. QEMU Log File
If you've started QEMU with logging options, you can also check the log files to retrieve information about the hardware and devices being used in the VM.

For example, starting QEMU with -D logfile.txt will create a log file with information about the devices and hardware events.

```
qemu-system-x86_64 -D logfile.txt
```

Summary of Commands and Tools:
QEMU Monitor Commands:

info qtree (device tree)
info pci (PCI devices)
info usb (USB devices)
info block (block devices)
info network (network interfaces)
Inside Guest OS (Linux):

lspci (PCI devices)
lsusb (USB devices)
lsblk (block devices)
dmesg (kernel messages)
/proc/cpuinfo and /proc/meminfo (CPU/memory info)
By using these methods, you can effectively inspect and retrieve detailed information about the devices used by a particular QEMU virtual machine during runtime.

![dump](https://raw.githubusercontent.com/KarolDuracz/scratchpad/refs/heads/main/bootloader_x86/cdrom_int13/118%20-%2029-09-2024%20-%20cd.png)

<hr>
<h2>Ok, I have some information about virtual hardware and CDROM device. Write bootloader once again...</h2>

<b>Approach #1</b> <br /><br />
To write a simple bootloader for x86 using BIOS interrupt INT 13h to read data from a CD-ROM, you need to perform the following steps:

Set the video mode: Usually, we don't need to set the video mode manually, as BIOS does this, but depending on your situation, you might want to switch modes.

Use BIOS interrupts (INT 13h): This interrupt is used for low-level disk I/O, including CD-ROMs. Specifically, the INT 13h service for reading sectors (AH=02h) will help us read from a bootable CD.

Load the Bootloader: Use the BIOS to read sectors from the CD and load the second stage bootloader or operating system into memory.

CD-ROM Bootloader Outline:
We will use El Torito bootable CD specification (used for booting from a CD-ROM).
El Torito allows a CD-ROM to emulate a floppy disk, hard disk, or another bootable media.
Steps for Reading Data from CD-ROM Using INT 13h
1. Set Up Registers for INT 13h:
AH = 42h: BIOS Extended Read.
DL: Drive number of the CD-ROM (likely 0xE0 for the first CD-ROM).
ES:BX: Pointer to the disk address packet (DAP).
The Disk Address Packet (DAP) structure is as follows:

```
Disk_Address_Packet:
    db 0x10            ; Size of the structure (16 bytes)
    db 0x00            ; Reserved (must be 0)
    dw NumberOfSectors  ; Number of sectors to read
    dw BufferOffset     ; Offset in memory where data will be read
    dw BufferSegment    ; Segment in memory where data will be read
    dq LBA              ; Starting LBA (Logical Block Address)
```

2. Call INT 13h to Read from CD-ROM:

```
mov ah, 0x42           ; BIOS Extended Read
mov dl, 0xE0           ; CD-ROM drive number (0xE0 for the first CD-ROM)
mov si, Disk_Address_Packet
int 0x13               ; Call BIOS
jc error               ; Check for carry flag, if set, an error occurred
```

Example: Bootloader Code in Assembly

```
[bits 16]              ; Real mode (16-bit code)
[org 0x7C00]           ; BIOS loads bootloader here in real mode

start:
    ; Clear registers
    xor ax, ax
    xor bx, bx
    xor cx, cx
    xor dx, dx

    ; Display message (optional)
    mov si, boot_message
    call print_string

    ; Prepare disk address packet for INT 13h, AH = 42h (BIOS Extended Read)
    mov ax, 0x07C0      ; Load to 0x07C0:0000 (right after bootloader in memory)
    mov es, ax
    mov bx, 0x0000      ; Buffer offset

    ; Prepare disk address packet
    mov byte [DAP], 0x10       ; Size of DAP (16 bytes)
    mov byte [DAP+1], 0x00     ; Reserved (0)
    mov word [DAP+2], 1        ; Number of sectors to read
    mov word [DAP+4], 0x0000   ; Buffer offset in memory (0000)
    mov word [DAP+6], 0x07C0   ; Buffer segment in memory (07C0)
    mov dword [DAP+8], 0x10    ; Starting LBA (Logical Block Address)

    ; Call BIOS to read sector from CD-ROM
    mov ah, 0x42        ; BIOS Extended Read
    mov dl, 0xE0        ; CD-ROM drive number (0xE0 for first CD-ROM)
    mov si, DAP         ; Pointer to Disk Address Packet (DAP)
    int 0x13            ; Call BIOS

    jc error            ; Jump to error if carry flag is set

    ; Continue loading the bootloader from CD (if successful)
    ; Jump to the loaded code (e.g., at 0x07C0:0000)
    jmp 0x07C0:0000

error:
    ; Handle read error (optional)
    mov si, error_message
    call print_string
    cli
    hlt

; Print a string (terminated by '$')
print_string:
    mov ah, 0x0E
.next_char:
    lodsb
    cmp al, '$'
    je .done
    int 0x10
    jmp .next_char
.done:
    ret

; Data
boot_message db "Booting from CD-ROM...$", 0
error_message db "Error reading CD-ROM!$", 0

; Disk Address Packet (DAP) structure
DAP:
    db 0x10, 0x00       ; DAP size and reserved byte
    dw 1                ; Number of sectors to read
    dw 0x0000           ; Buffer offset
    dw 0x07C0           ; Buffer segment
    dq 0x10             ; Starting LBA (Logical Block Address)

times 510-($-$$) db 0   ; Padding to 510 bytes
dw 0xAA55               ; Boot signature (0xAA55)
```

Step-by-Step Breakdown:
Clear Registers: The code clears the registers at the beginning, making sure there’s no leftover data from earlier.

Display Boot Message: This optional part prints a string (Booting from CD-ROM...) on the screen using BIOS interrupt INT 10h, which handles basic text output in real mode.

Prepare Disk Address Packet: This is the most critical part:

The DAP structure is created in memory with details like the number of sectors to read (1 sector in this case), buffer segment, offset, and the starting LBA (Logical Block Address) of the data to read.
Here, we set LBA = 0x10, which is a common starting point for bootable CD-ROM sectors (depends on your setup).
Call INT 13h to Read Data:

Registers are set for INT 13h (Extended Read, AH=42h) and the CD-ROM drive number (DL=0xE0).
The disk address packet is passed using the SI register, and INT 13h is called to load the sector into memory at address 0x07C0:0000.
Error Handling: If an error occurs (carry flag is set), the bootloader will display an error message and halt the system.

Jump to Loaded Code: If the sector read is successful, the code jumps to the loaded bootloader code, which resides at 0x07C0:0000.

Key Points:
Sectors and LBAs: On CD-ROMs, data is usually read in sectors (typically 2048 bytes for ISO-9660 CDs), and sectors are addressed using the Logical Block Address (LBA). The BIOS reads the CD-ROM in units of sectors.

INT 13h Extended Read: The function INT 13h, AH=42h is part of the BIOS Extensions, which support reading from hard drives and CD-ROMs using logical block addressing (LBA) rather than Cylinder-Head-Sector (CHS) addressing.

This bootloader reads 1 sector from the CD-ROM and jumps to it, ready to load the next stage (e.g., a kernel or a more advanced bootloader).
<br /><br />
<b>Approach #2</b> <br /><br />
To write a bootloader that loads a kernel from the ide1-cd0 device (which is a CD-ROM connected via IDE in QEMU) using BIOS interrupts (INT 13h), we will go step-by-step through the bootloading process. This bootloader will:

Use BIOS interrupts to initialize and load sectors from the CD-ROM (INT 13h).
Detect the correct device (i.e., ide1-cd0) and load the kernel from the boot sector.
Jump to the kernel's entry point.
Step 1: Understanding INT 13h for CD-ROM Access
BIOS provides a series of disk services through INT 13h. CD-ROMs connected via IDE are accessed as drives in the BIOS, starting from drive number 0x80 (first hard drive) or 0x00 (first floppy).

For CD-ROMs specifically, when accessed through INT 13h, you need to understand how to:

Identify the CD-ROM using BIOS disk services.
Read sectors from the CD-ROM using INT 13h.
BIOS INT 13h Functions:
INT 13h, AH = 42h (Extended Read): This function is used to read from large drives (including CD-ROMs).
INT 13h, AH = 15h (Drive Type Check): Check the type of device connected.
INT 13h, AH = 41h: Check if BIOS supports extensions for reading CD-ROMs.
Step 2: Writing the Bootloader
The bootloader must fit in the first 512 bytes (boot sector). This example assumes you have a bootable ISO (boot_cs_1.iso) and want to load the kernel stored on that ISO.

Here’s a simple x86 assembly code for a bootloader that:

Initializes the system.
Uses INT 13h to read sectors from ide1-cd0.
Loads the kernel from the ISO into memory.

```
BITS 16                   ; We are in 16-bit real mode
ORG 0x7C00                ; Boot sector loads at 0x7C00 by BIOS

start:
    ; Clear registers
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00         ; Set stack pointer

    ; Print message to screen (for debugging)
    mov si, boot_msg
    call print_string

    ; Select the drive (0x01 for `ide1-cd0`, as per QEMU config)
    mov dl, 0x01           ; DL = 0x01 for secondary IDE (CD-ROM)

    ; Load kernel from CD-ROM (assume it's at LBA 0x10 on CD)
    mov bx, 0x9000         ; BX = Load the kernel at 0x9000 (arbitrary memory location)
    mov si, 0x10           ; LBA = 0x10 (sector to load from)
    call read_sector       ; Call read sector function

    ; Jump to loaded kernel
    jmp 0x9000             ; Jump to where we loaded the kernel

hang:
    jmp hang               ; If something goes wrong, loop forever

; Function: read_sector
; Reads one sector (512 bytes) from the CD-ROM (using LBA addressing)
read_sector:
    push ax
    push bx
    push cx
    push dx

    ; LBA to CHS conversion (INT 13h uses CHS, not LBA)
    mov ah, 0x02           ; AH = 0x02 (Read Sector)
    mov al, 0x01           ; AL = Number of sectors to read (1 sector = 512 bytes)
    mov ch, 0x00           ; CH = Cylinder number (0)
    mov cl, si             ; CL = Sector number (here we use LBA as sector directly for simplicity)
    mov dh, 0x00           ; DH = Head number (0)

    int 0x13               ; Call BIOS Disk Service to read sector

    jc read_error          ; If error (Carry Flag set), jump to error handler

    ; Sector read successfully
    pop dx
    pop cx
    pop bx
    pop ax
    ret                    ; Return to caller

read_error:
    ; Print error message and halt the system
    mov si, error_msg
    call print_string
    jmp hang

; Function: print_string
; Print a null-terminated string pointed to by SI to the screen
print_string:
    mov ah, 0x0E           ; BIOS teletype output
.print_char:
    lodsb                  ; Load byte at [SI] into AL
    cmp al, 0               ; Check if it's the null terminator
    je .done
    int 0x10               ; BIOS interrupt to print character in AL
    jmp .print_char
.done:
    ret

boot_msg db 'Booting kernel from CD-ROM...', 0
error_msg db 'Error reading from CD-ROM!', 0

TIMES 510-($-$$) db 0      ; Pad boot sector with zeros up to 510 bytes
DW 0xAA55                  ; Boot signature (0x55AA)
```

Step 3: Explanation of the Bootloader
Stack Initialization: The bootloader sets up the stack to avoid potential issues when the kernel starts.

Print a Message: The bootloader prints a message indicating that it is starting the kernel load process.

Select the Device: The dl register is set to 0x01 to select the secondary IDE device (ide1-cd0). QEMU assigns dl values based on the drive number:

0x00: First floppy
0x80: First hard drive
0x01: Second IDE device, which is the CD-ROM in this case (ide1-cd0).
LBA Addressing: The bootloader loads the kernel from the CD-ROM using Logical Block Addressing (LBA). We assume the kernel starts at sector 0x10 on the CD (you can adjust this based on the actual location of the kernel on your ISO).

Reading Sectors via INT 13h: The bootloader uses the BIOS INT 13h, AH = 02h function to read the sector from the CD-ROM into memory. The sector is loaded at address 0x9000.

Jump to the Kernel: After loading the kernel into memory, the bootloader jumps to the kernel's entry point at 0x9000.

Error Handling: If reading from the CD-ROM fails, the bootloader prints an error message and halts the system.

Step 4: Create the Bootable ISO
Once you have the bootloader assembled and the kernel ready, you need to create a bootable ISO image containing the bootloader and kernel.

Assemble the Bootloader: Use an assembler like nasm to assemble the bootloader:

```
nasm -f bin bootloader.asm -o bootloader.bin
```

Create the Bootable ISO: Use genisoimage (or mkisofs) to create the ISO:

```
genisoimage -b bootloader.bin -o boot_cs_1.iso /path/to/kernel
```

-b bootloader.bin: Specifies the bootloader file.
/path/to/kernel: Path to the kernel you want to include in the ISO.
Step 5: Boot the ISO in QEMU
Now that you have the bootable ISO, you can boot it in QEMU:

```
qemu-system-x86_64 -cdrom boot_cs_1.iso -boot d
```

This will boot from the CD-ROM (ide1-cd0), load the bootloader, and the bootloader will load the kernel from the CD-ROM into memory.

Summary
The bootloader uses BIOS INT 13h to access the CD-ROM.
It reads the kernel from the specified sector (LBA).
The bootloader jumps to the kernel after loading it into memory.
The ISO is created with the bootloader and kernel, and you boot it using QEMU.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/cdrom_int13/119%20-%2029-09-2024%20-%20cd.png?raw=true)

But line ```mov cl, si             ; CL = Sector number (here we use LBA as sector directly for simplicity)``` we need change <br />
 In my original code, the instruction mov cl, si is incorrect because SI is a 16-bit register, and CL is only an 8-bit register, meaning this would truncate the value of SI. We need to properly split the 16-bit LBA value (Logical Block Address) into appropriate components to work with CHS (Cylinder, Head, Sector) addressing, as BIOS INT 13h does not directly support LBA and uses CHS (Cylinder, Head, Sector) addressing instead.

To handle this properly, we need to calculate Cylinder, Head, and Sector from the 16-bit LBA value and then load them into the appropriate registers.

Correcting the Code for LBA to CHS Translation
To fix this issue, here’s a better way to convert a 16-bit LBA to CHS format, assuming that the CD-ROM uses 63 sectors per track (which is typical for hard drives and many BIOS configurations).

The CHS calculation works as follows:

Cylinder: High part of the LBA
Head: Number of heads (often 0 or 1 for CDs)
Sector: Must be in the range 1–63 (BIOS INT 13h expects this in 1-based format)
Fixed Bootloader Code with LBA to CHS Conversion
Here’s an improved version of the code with the correct LBA to CHS conversion:

```
BITS 16                   ; We are in 16-bit real mode
ORG 0x7C00                ; Boot sector loads at 0x7C00 by BIOS

start:
    ; Clear registers
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00         ; Set stack pointer

    ; Print message to screen (for debugging)
    mov si, boot_msg
    call print_string

    ; Select the drive (0x01 for `ide1-cd0`, as per QEMU config)
    mov dl, 0x01           ; DL = 0x01 for secondary IDE (CD-ROM)

    ; Load kernel from CD-ROM (assume it's at LBA 0x10 on CD)
    mov bx, 0x9000         ; BX = Load the kernel at 0x9000 (arbitrary memory location)
    mov si, 0x10           ; SI = LBA = 0x10 (sector to load from)
    call read_sector       ; Call read sector function

    ; Jump to loaded kernel
    jmp 0x9000             ; Jump to where we loaded the kernel

hang:
    jmp hang               ; If something goes wrong, loop forever

; Function: read_sector
; Reads one sector (512 bytes) from the CD-ROM (using LBA addressing)
read_sector:
    push ax
    push bx
    push cx
    push dx

    ; Convert LBA (in SI) to CHS for INT 13h
    mov ax, si              ; AX = LBA
    xor dx, dx              ; DX = 0 (clear upper bits)
    div WORD [sectors_per_track] ; AX = Cylinder, DX = Sector (1-based)

    ; Load CHS values into the correct registers
    mov ch, al              ; Cylinder (lower byte into CH)
    xor dh, dh              ; DH = Head (0 for CD-ROM)
    inc dl                  ; Convert DX (sector) to 1-based
    mov cl, dl              ; CL = Sector number

    ; BIOS INT 13h - Read sector
    mov ah, 0x02            ; AH = 0x02 (Read Sector)
    mov al, 0x01            ; AL = Number of sectors to read (1 sector = 512 bytes)
    int 0x13                ; Call BIOS Disk Service to read sector

    jc read_error           ; If error (Carry Flag set), jump to error handler

    ; Sector read successfully
    pop dx
    pop cx
    pop bx
    pop ax
    ret                     ; Return to caller

read_error:
    ; Print error message and halt the system
    mov si, error_msg
    call print_string
    jmp hang

; Function: print_string
; Print a null-terminated string pointed to by SI to the screen
print_string:
    mov ah, 0x0E           ; BIOS teletype output
.print_char:
    lodsb                  ; Load byte at [SI] into AL
    cmp al, 0               ; Check if it's the null terminator
    je .done
    int 0x10               ; BIOS interrupt to print character in AL
    jmp .print_char
.done:
    ret

boot_msg db 'Booting kernel from CD-ROM...', 0
error_msg db 'Error reading from CD-ROM!', 0

sectors_per_track dw 63    ; Common value for sectors per track

TIMES 510-($-$$) db 0      ; Pad boot sector with zeros up to 510 bytes
DW 0xAA55                  ; Boot signature (0x55AA)
```

Key Changes:
LBA to CHS Conversion:

LBA is stored in SI.
The div instruction divides AX (which holds the LBA) by the number of sectors per track (in this case, 63). This gives the Cylinder and Sector values.
The quotient (AX) becomes the Cylinder.
The remainder (DX) gives the Sector (which must be incremented by 1, as BIOS uses 1-based sector numbering).
The Head is set to 0 (for simplicity, as CDs usually have only one head per track).
Register Use:

CH holds the Cylinder (calculated from LBA).
DH is set to 0 for the Head.
CL holds the Sector (1-based).
Sector per Track:

In the CD-ROM's case, it often uses the standard 63 sectors per track, which is common in CHS BIOS addressing schemes for hard drives and CD-ROMs.
Explanation:
The CHS values are derived by dividing the LBA by the number of sectors per track:
The Cylinder is the quotient.
The Sector is the remainder (adjusted to be 1-based).
The Head is assumed to be 0 for simplicity.
We load these values into the appropriate registers (CH, CL, and DH) and use INT 13h to read a sector from the CD-ROM.
Step-by-Step Process for Reading Data:
Initialize registers: Clear the registers, and set up the stack.
Print the boot message: Just for debugging, we print a message saying we are booting the kernel.
Select the CD-ROM device: We use DL = 0x01 to select ide1-cd0 (secondary IDE device, typically the CD-ROM in this setup).
LBA to CHS conversion: The bootloader converts the Logical Block Address (SI = 0x10) to Cylinder, Head, and Sector using simple division.
Read the kernel: The bootloader uses BIOS INT 13h to load the sector containing the kernel into memory (BX = 0x9000).
Jump to the kernel: After successfully loading the kernel, the bootloader jumps to the loaded kernel's memory address (0x9000).
Error handling: If any error occurs during sector reading, an error message is displayed and the system halts.
Conclusion:
This fixed bootloader properly handles the LBA to CHS conversion and reads a sector from the CD-ROM (ide1-cd0). You can further modify it based on where your kernel is located on the ISO image and how large the kernel is (i.e., you might need to read multiple sectors).

Look like something works but print some errors. Run from this command: <br />

```
qemu-system-x86_64 -cdrom boot_cd_2.iso -boot d -m 64 -d int
```

<br /><br />
<b>Approach #3</b> <br /><br />
"Error reading from CD-ROM!", this suggests that the INT 13h BIOS call to read the sector failed. The cause could be related to one of the following issues:

Invalid LBA/CHS Values: If the LBA to CHS conversion is incorrect, or if the target sector is out of range for the CD-ROM, the BIOS will fail to read the data.
Incorrect Drive: Ensure that the correct drive number is used (DL = 0x01 for ide1-cd0 in QEMU).
CD-ROM Configuration in QEMU: Verify that the CD-ROM is properly configured in QEMU and the ISO is accessible.
INT 13h Errors: The BIOS could return an error, and you may not be handling it correctly.
Let's walk through a few debugging steps to troubleshoot this issue.

Debugging Steps
Check the Carry Flag (CF): The BIOS will set the Carry Flag (CF) if there's an error during the INT 13h call. After the INT 13h call, check if the carry flag is set and output the error code returned in the AH register to identify the problem.

Verify LBA to CHS Conversion: Ensure that the LBA to CHS conversion is correct and doesn't result in invalid CHS values that might be out of bounds for the CD-ROM. For example, make sure that the Sector is between 1 and 63 (inclusive).

Output the BIOS Error Code: Print the BIOS error code from AH to help identify the issue. Here's how you can modify the error handling to show the error code.

Improved Error Handling
Add an error code printout to understand the specific issue during the INT 13h read:

```
read_error:
    ; Print error message
    mov si, error_msg
    call print_string

    ; Output error code (in AH) as a hexadecimal value
    mov ah, 0x00            ; Reset AH to clear previous content
    mov al, ah              ; Move error code from AH to AL
    call print_hex           ; Print the error code in hexadecimal format

    jmp hang

; Function: print_hex
; Prints the value in AL as a 2-digit hexadecimal number
print_hex:
    push ax
    push bx

    ; Print high nibble
    mov bl, al
    shr al, 4               ; Shift the high nibble into the lower 4 bits
    call print_nibble

    ; Print low nibble
    mov al, bl
    and al, 0x0F            ; Mask out the upper nibble
    call print_nibble

    pop bx
    pop ax
    ret

; Function: print_nibble
; Prints the nibble (low 4 bits) of AL as a hexadecimal digit
print_nibble:
    add al, '0'             ; Convert nibble to ASCII
    cmp al, '9'             ; If it's greater than '9', adjust for letters
    jle .print_char
    add al, 7               ; Adjust ASCII for 'A' through 'F'

.print_char:
    mov ah, 0x0E            ; BIOS teletype output
    int 0x10                ; Print the character in AL
    ret
```

Explanation of Changes:
Print Error Code:

When an error occurs during the INT 13h call, the BIOS returns an error code in the AH register.
This code will print the error message along with the error code as a two-digit hexadecimal value, allowing you to see exactly what error occurred.
Handling Nibbles:

```
print_hex splits the byte in AL into two 4-bit nibbles (the high and low parts of the byte), and prints them as hexadecimal digits.
BIOS Error Codes from INT 13h:
0x00: No error.
0x01: Invalid function or parameter.
0x02: Address mark not found.
0x03: Disk write-protected.
0x04: Sector not found (this can occur if the CHS values are wrong).
0x05: Reset failed.
0x10: CRC error during data transfer.
0x20: Controller failure.
0x40: Seek failure.
0x80: Timeout error (drive not ready).
0xAA: Drive not ready.
```

Possible Fixes Based on the Error Code:
Error 0x04 (Sector Not Found): This could indicate that the LBA to CHS conversion resulted in an invalid CHS address (such as a sector number out of range).

Double-check the LBA value and conversion logic to ensure you're reading from the correct sector.
Verify that the kernel is located at the correct sector in the ISO.
Error 0x80 (Drive Not Ready): This could indicate that the CD-ROM is not accessible or not ready.

Ensure that the QEMU configuration properly mounts the ISO and that the CD-ROM is available as ide1-cd0.
Error 0x01 (Invalid Function/Parameter): This could indicate an issue with the INT 13h parameters, such as passing an invalid drive number or using incorrect CHS values.

Further Steps:
If you're still having issues after obtaining the error code, we can further investigate based on the specific BIOS error, or adjust the QEMU configuration to ensure the CD-ROM device is correctly initialized.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/cdrom_int13/120%20-%2029-09-2024%20-%20cd.png?raw=true)

Again. Somethings work but... probably I need to learn more about qemu and CTRL + ALT + 2 to debug what's going on etc.
<br /><br />
THIS FILES boot_cs_1, boot_cs_2, boot_cs_3 is there in this repo !

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/cdrom_int13/121%20-%2029-09-2024%20-%20finally%20on%20vm%20it%20works.png?raw=true)

Finally it works on Virtualbox 7 too ;p
<br /><br />
update 29-09-2024 : OK. That's it. That's it for now. Over the course of 2 days this weekend I've refreshed this topic a bit. But from what I can see, there's a ton of learning to get through it. It's probably pointless. Just yesterday I thought that learning this was a "priority" but it's really a huge amount of knowledge to learn to have the basics. I'm struggling at the moment. But I've taken a step further than last time when I started learning how to write a bootloader and hello world x86.
