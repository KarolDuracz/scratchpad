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


