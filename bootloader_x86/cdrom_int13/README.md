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


