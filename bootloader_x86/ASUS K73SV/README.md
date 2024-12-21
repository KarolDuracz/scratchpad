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
[2] https://www.alldatasheet.com/datasheet-pdf/pdf/1995319/ITE/IT8502E.html - not exactly U3001 (IT8572E) but pinout is similar  <br />
[3] https://www.youtube.com/watch?v=toCDtDdd2oU&ab_channel=FIXstudio - disassemble laptop asus<br />
[4] https://www.datasheets360.com/part/detail/asm1442/-7813675855283622004/ -  C.S ASM1442 QFN-48 ASMEDIA hdmi controller
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

HM65 and some pins to DVI connector. I haven't checked the connections thoroughly, but I think it goes through the HDMI controller. Some pins to control R G B VSYNC HSYNC (PCH)
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/ASUS%20K73SV/board%20view%20pics/215%20-%2021-12-2024%20-%20cd.png?raw=true)

Bottom side 
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/ASUS%20K73SV/Plyta-glowna-Asus-X73S-K73SJ%20-%20Copy.jpg?raw=true)

quick look at dump via EFITool bios and vbios
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/ASUS%20K73SV/board%20view%20pics/bios%20and%20vbios.png?raw=true)

