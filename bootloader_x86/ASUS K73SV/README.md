<h3>Placing this here is for educational purposes only. To have a rough idea of ​​what it looks like on real hardware.
</h3>
<hr>
<h2>"backup" for review and learn sth about it - this is not original firmware from my laptop - only for tests and research</h2>
bios firmware [K73SVAS.209]<br />
vrom [nvidia GT540M vbios.rom]<br />
dump dsdt.dsl <br />
<h2>Tools</h2>

BoardViewer 2.0.1.9 beta http://boardviewer.net/
<br /><br />
Uefitool ne alpha A68 https://github.com/LongSoft/UEFITool/releases
<br /><br />
iASL Compiler and Windows ACPI Tools (.zip, 1.3 MB) toolchain - Get https://www.intel.com/content/www/us/en/developer/topic-technology/open/acpica/download.html 
<hr >
[1] https://www.alldatasheet.pl/datasheet-pdf/pdf/1178999/INTEL/BD82HM65.html - HM 65 - windows driver <br />
[1a] https://www.intel.com/content/dam/www/public/us/en/documents/datasheets/6-chipset-c200-chipset-datasheet.pdf - 6-chipset-c200-chipset<br />
[2] https://www.alldatasheet.com/datasheet-pdf/pdf/1995319/ITE/IT8502E.html - not exactly U3001 (IT8572E) but pinout is similar  [ the common name is probably also "KBC" ] <br />
[3] https://www.youtube.com/watch?v=toCDtDdd2oU&ab_channel=FIXstudio - disassemble laptop asus<br />
[4] https://www.datasheets360.com/part/detail/asm1442/-7813675855283622004/ -  C.S ASM1442 QFN-48 ASMEDIA hdmi controller<br />
[5] https://www.alldatasheet.com/datasheet-pdf/pdf/575462/MCNIX/25L3206E.html - BIOS flash chip<br />
[6] https://www.rom.by/files/uP7706.pdf - the chip next to KBC and bios flash to know that it is not a bios chip<br />
[7] https://www.datasheetcafe.com/it8572e-datasheet-pdf/ - it8572e
<hr>
<h2>Quick look on motherboard ASUS K73SV</h2>

HM65 datasheet | For Windows Device Manager > BIOS Device Name : <b>\_SB.PCI0.SBRG</b> | Look into dsdt.dsl
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/ASUS%20K73SV/board%20view%20pics/213%20-%2021-12-2024%20-%20no%20jest%20HM65.png?raw=true)

Top side motherboard and
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/ASUS%20K73SV/board%20view%20pics/216%20-%2021-19-2024%20-%20no%20i%20jest.png?raw=true)

Connectors to touchpad and keyboard
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/ASUS%20K73SV/board%20view%20pics/217%20-%2021-19-2024%20-%20cd.png?raw=true)

For example keyboard connector is attached to U3001 and pins KSOxx etc
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/ASUS%20K73SV/board%20view%20pics/218%20-%2021-19-2024%20-%20podobny%20uklad.png?raw=true)

This ITE chip (U3001) is connected one of the pin CPU. PECI. Platform Environment Control Interface (PECI) - is a single-wire communication interface used by Intel CPUs to facilitate thermal and power management. The PECI pin on an Intel CPU allows external devices, like the motherboard chipset or Baseboard Management Controller (BMC), to communicate with the CPU for real-time monitoring and control.
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/ASUS%20K73SV/board%20view%20pics/219%20-%2021-12-2024%20-%20cd.png?raw=true)

This make sense, because U3001 is attached to FAN connector for CPU cooling
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/ASUS%20K73SV/board%20view%20pics/220%20-%2021-12-2024%20-%20cd.png?raw=true)

All PECI connecitons - I choose this, that was the first thing I started analyzing for this quick intro. But you can download "Asus K73SJ Rev2.4 Boardview(FZ).fz" file and BoardViewer to examine all connections.
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/ASUS%20K73SV/board%20view%20pics/232%20-%2023-12-2024%20-%20PECI.png?raw=true)

HM65 and some pins to DVI connector. I haven't checked the connections thoroughly, but I think it goes through the HDMI controller. Some pins to control R G B VSYNC HSYNC (PCH)
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/ASUS%20K73SV/board%20view%20pics/215%20-%2021-12-2024%20-%20cd.png?raw=true)

Bottom side 
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/ASUS%20K73SV/Plyta-glowna-Asus-X73S-K73SJ%20-%20Copy.jpg?raw=true)

quick look at dump via EFITool bios and vbios
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/ASUS%20K73SV/board%20view%20pics/bios%20and%20vbios.png?raw=true)

BIOS flash chip (probably) - I took the datasheet of another system which is similar but after closer look I can see that it is probably WINBOND, like on here https://expressit.pl/porady-komputerowe/programowanie-ukladu-kbc-w-laptopie-wgrywanie-wsadu-ec-kbc-programator-kbc/  <br />
https://www.mouser.pl/ProductDetail/Winbond/W25X10CLSNIG?qs=qSfuJ%252Bfl%2Fd58%252Bc3wxS8Vmg%3D%3D&srsltid=AfmBOooJJhTgmUqUAIzapoC7VdQg5TwAGzVESEX518Wao65tPLrEGiMT <br /> https://www.mouser.pl/datasheet/2/949/w25x10cl_revg_021714-1489755.pdf <br /> But pinout is similar . Pins 2 and 5 is for data In/Out. WP is on 3. GND 4, VCC 8, so looks the same.
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/ASUS%20K73SV/board%20view%20pics/223%20-%2022-12-2024%20-%20bios%20chip.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/ASUS%20K73SV/board%20view%20pics/229%20-%2022-12-2024%20-%20winbond.png?raw=true)

Another chip next to KBC and BIOS flash - uP7706
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/ASUS%20K73SV/board%20view%20pics/224%20-%2022-12-2024%20-%20bios%20chip%20%232.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/ASUS%20K73SV/board%20view%20pics/225%20-%2022-12-2024%20-%20cd.png?raw=true)

Pin 5 from BIOS chip is onnected to PIN 102 (U3001) - SI_SEC
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/ASUS%20K73SV/board%20view%20pics/226%20-%2022-12-2024%20-%20cd.png?raw=true)

Pin 2 from BIOS connected to 103 on KBC (EC_SO_PCH)
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/ASUS%20K73SV/board%20view%20pics/227%20-%2022-12-2024%20-%20cd.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/ASUS%20K73SV/board%20view%20pics/228%20-%2022-12-2024%20-%20cd.png?raw=true)

KBC layout pinout - it8572e model <br />
Look at right top corner to find 117 and 118 pins <br />
PIN 117 and 118 is connected to PECI. <br />
118 pin --> Q2001 (N-MOSFET 2N7002         SOT-23 PHILIPS) pin 1 --> this philips from pin 3 to R2043 in pin 2 --> And from this pin 2 to U2001 ( BD82HM65 PCH) PIN HDA_SDO ACZ_SDOUT<br />
117 pin --> U3001 to R3007 ------> TO U0301 this means direct to CPU chip to pin PECI (peci_ec)
<br /><br />
And in other hand. Pin 103 from previous picture 228.xxxxxxxxx  is connected to SPI. Exactly to pin 103. Look at KBC schema which is below.
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/ASUS%20K73SV/board%20view%20pics/IT8572E-Datasheet.jpg?raw=true)

Explain for this BIOS chip.  Because typical information about memory layout looks like that. Because I still don't know at what point this K73SVAS.209 is loaded. From the UEFI standard it is known that for Windows it looks for EFI\Boot\bootx64.efi but where do these strange memory regions that are outside the EFI areas come from? (0x00100000 - 0x7FFFFFFF: Available memory for EFI applications, OS loaders, and boot services.) - One answer is that it is loaded from external FLASH SPI memory. If you look at the memory region of this EFITool dump which is above it has the following addresses FFF70000, FFF30000, FFE10000, i.e. beyond what the guide says.

```
Memory Layout Explanation
0x00000000 - 0x000FFFFF: Reserved for Low-Memory Operations

0x00000000 - 0x000003FF: Interrupt Vector Table (IVT)
Contains 256 interrupt vectors, each 4 bytes, used in real-mode operations.
0x00000400 - 0x000004FF: BIOS Data Area (BDA)
Stores information about hardware resources like disk drives and COM ports.
0x00000500 - 0x00007BFF: Free or OEM Reserved
Often used by firmware or custom hardware during initialization.
0x00007C00 - 0x00007DFF: Legacy Bootloader Area
The location where legacy bootloaders (e.g., MBR) are loaded.
0x00007E00 - 0x0009FFFF: Conventional Memory
Used for temporary data during boot, like stack and buffers.
0x000A0000 - 0x000BFFFF: Video Memory
Memory-mapped I/O for text or graphics modes (VGA frame buffer).
0x000C0000 - 0x000FFFFF: BIOS ROM
Contains firmware or shadowed BIOS code copied from ROM to RAM for faster execution.
0x00100000 - 0x7FFFFFFF: Available Memory

0x00100000 - 0x002FFFFF: Reserved for Boot Services
Firmware allocates this region for boot services and firmware internal operations.
0x00300000 - 0x7FFFFFFF: Available for UEFI Applications, OS Loaders, and Memory Allocation
This region is dynamically managed by the UEFI memory map and allocated for:
EFI application code and data.
Bootloader code (e.g., GRUB or Windows Boot Manager).
Stack, heap, and runtime memory for the UEFI environment.
```

```
 many key components such as the EFI Firmware File System (FFS) structures reside in high memory ranges that are mapped to the SPI flash chip (firmware storage). These high memory addresses, such as 0xFFF70000, 0xFFF30000, and 0xFFE10000, correspond to firmware regions that are mapped into the CPU's addressable space during execution.

Why High Memory Addresses Are Used
Memory-Mapped SPI Flash:

Modern systems map the SPI flash (containing firmware) to high physical memory addresses in the CPU's address space.
These addresses (e.g., 0xFFF70000) are typically reserved for firmware execution and are distinct from the RAM available to EFI applications.
Firmware Storage Layout:

The firmware is segmented into regions for various purposes:
Boot Block.
BIOS Region.
Embedded Controllers.
NVRAM Storage.
Firmware File System (FFS).
Tools like UEFITool reveal these segments in the firmware binary.
Memory Map and Access:

During execution, the UEFI firmware maps portions of the SPI flash into memory.
These mapped regions are accessible to the CPU, but they are not part of the general-purpose memory available to EFI applications or the operating system.
Memory Layout Including High Firmware Regions
Here’s an updated memory layout that incorporates these firmware-mapped regions:

General Memory Layout
Low Memory (< 1MB):

Reserved for legacy compatibility and low-level system structures (IVT, BDA, etc.).
Main System Memory (1MB to 2GB or higher):

Used by EFI boot services, EFI runtime services, and operating systems.
High Reserved Memory (Firmware Regions):

Firmware SPI flash memory is mapped to specific high addresses, typically:
0xFFF00000 - 0xFFFFFFFF:
Contains the firmware, including the EFI Firmware File System.
This region is not writable or modifiable by applications and is often marked as "Reserved" in the memory map.
Other high reserved regions for ACPI tables, SMBIOS, or other firmware data.
Firmware SPI Flash Mapping
FFS GUID Areas:

The EfiFirmwareFileSystemGUID regions (0xFFF70000, 0xFFF30000, 0xFFE10000) correspond to portions of the SPI flash where firmware components are stored. These regions typically include:
Boot services drivers.
Runtime services drivers.
Configuration data (e.g., microcode, platform-specific settings).
Execution Context:

The firmware code in these regions may be directly executed in place (XIP) or copied into RAM for execution, depending on the system.
Why Does the EFI Memory Map Show RAM Limits at 0x7FFFFFFF?
The UEFI memory map provided to EFI applications and the OS only shows usable system RAM and reserves regions for specific purposes (e.g., ACPI, firmware, I/O).

RAM vs. Reserved Regions:

The range 0x00100000 to 0x7FFFFFFF typically represents the usable DRAM for applications and services.
High memory regions like 0xFFF00000 are reserved for firmware and are not part of the general-purpose memory map.
Reserved Regions:

The firmware maps SPI flash regions (like 0xFFF70000) but marks them as Reserved Memory in the EFI memory map.
How EFI Applications and OS Loaders Handle This
Firmware Access:

EFI applications and OS loaders access firmware regions (e.g., NVRAM, drivers) using UEFI-provided runtime services.
Direct access to 0xFFF70000 is not expected unless explicitly handled by the firmware.
Mapped Memory:

The EFI system ensures that any required data (e.g., firmware drivers or configuration) is either available in the memory map or accessible through specific UEFI APIs.
Summary
The ranges 0xFFF70000, 0xFFF30000, etc., represent high memory regions mapped from the SPI flash, which contain firmware components like the EFI Firmware File System.
These regions are distinct from the DRAM available to EFI applications (0x00100000 to 0x7FFFFFFF).
The discrepancy arises because the EFI memory map only exposes usable system memory, while high firmware regions are reserved and not accessible as general-purpose memory.
If you are examining these mappings with tools like UEFITool, you're looking at how the firmware is structured on the SPI flash, not the memory map provided to the operating system or applications.
```

```
Steps for Real-Mode to Long-Mode Transition
Start in Real Mode (16-bit)

At power-on, the CPU starts executing the reset vector (0xFFFF0).
The reset vector code typically contains a JMP instruction to a bootstrap routine in the firmware.
Switch to Protected Mode (32-bit)

Load the Global Descriptor Table (GDT) to define memory segments.
Set the CR0 register's PE (Protection Enable) bit to enter protected mode.
Update segment registers to point to appropriate descriptors in the GDT.
Initialize Memory Controller

Initialize DRAM and other hardware resources for full memory access.
Transition to Long Mode (64-bit)

Load a 64-bit-capable GDT with descriptors for flat memory addressing.
Enable PAE (Physical Address Extension) in the CR4 register.
Load the Page Tables for long-mode addressing.
Set the LME (Long Mode Enable) bit in the Extended Feature Enable Register (EFER).
Set the PG (Paging Enable) bit in CR0 to enable paging.
```

Some configuration for antother model of ASUS K73xx

```
Model: ASUS X73SV
Platforma: K73SD REV 2.3
CPU: Intel Core i5-2430M SR04W / Core i7-2670QM SR02N
GPU: nVidia N12P-GS-A1 (GT 540M)
PCH: Intel HM65 BD82HM65
KBC: ITE IT8572E
BIOS IC: 25L3206E (4MB)
```

<hr>
[1] https://opensecuritytraining.info/IntroBIOS_files/Day2_01_Advanced%20x86%20-%20BIOS%20and%20SMM%20Internals%20-%20SPI%20Flash.pdf - some info about SPI flash bios memory

<br /><br />
That's it for this quick intro to this topic.
