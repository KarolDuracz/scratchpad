<h2>Access to VGA Controller via WMD - related to EFI topic about VGA</h2>
related to this -> https://github.com/KarolDuracz/scratchpad/tree/main/bootloader_x86/tianocore%20EDK2/demo4%20-%20VGA%20controller
<br /><br />
I'm trying to understand if there is a correlation between the VGA controller and the GOP protocol in UEFI. But this is device for PCI line, Bus 0, Device 2, Fun 0 typically. And this have own registers for configuration. For my case this is documentation for that : Page 95 https://www.intel.com/content/dam/www/public/us/en/documents/datasheets/xeon-e3-1200v6-vol-2-datasheet.pdf - this is not exactly datasheet for that, but to compare with chip from 2011. I can't find it now, but it has similar offsets. The overall structure is similar.
<br /><br />
This is for Ivy Bridge (in my case is Sandy Bridge) datasheet for PCI Dev 2, fn 0 https://www.x.org/docs/intel/IVB/IHD_OS_Vol3_Part2.pdf
<br /><br />
Main goal for this demo5. Can Windows Kernel Driver (WMD) read VGA registers? The answer is : YES.
<br /><br />
And after switching resolution in Windows Display configuration, from 1600x900 to 1024x768. I stopped the driver and reloaded it after changing the resolution. Nothing happen on these registers. So... Probably this is not the way to go. But the VGA topic is definitely interesting, and worth spending more time on, because if GOP is not supported, and the VGA standard is supported, then there is no other option but to choose VGA. But from what I read, new systems, GPU systems no longer have support for VGA. And GOP itself is an extension of VESA. But that's roughly it.
<br /><br />
And I got exactly what I have in the EFI demo. Because I misread something there. In general, I don't know much about VGA itself now. I won't write in detail what I've learned in the meantime. But the basic thing is:
<h2>This comes from Chat GPT about VGA and question to explain what this configuration means.</h2>
VGA vs. VESA vs. GOP: Key Differences
<br />
Mode	Resolution Support	How It Works<br />
VGA Standard (Mode 0x12, etc.)	Up to 640×480	Uses VGA CRTC registers<br />
VESA BIOS Extensions (VBE 2.0/3.0)	800×600, 1024×768, etc.	Uses VESA graphics modes (INT 10h, AX=4F02h)<br />
UEFI GOP (Graphics Output Protocol)	800×600+, 1024×768+	Uses EFI framebuffer (direct memory mapping, no VGA registers)<br />
GOP’s Resolutions Depend on the GPU & Firmware<br />
Many modern GPUs do not support classic VGA registers at all.<br />
Instead, GOP initializes the framebuffer at a fixed resolution (often 800×600 or 1024×768).<br />
The UEFI firmware loads a driver that sets the resolution, bypassing old VGA modes.
<br /><br />
And what I read from documentation and some with some help CHAT GPT, that is <br />
VGA 720x400 Text Mode (Mode 0x03)<br />
Resolution	720×400<br />
Character Size	9×16 pixels<br />
Total Scanlines	447 (from CR06 + CR07 overflow)<br />
Visible Scanlines	400<br />
Horizontal Total	768 pixels (from CR00)<br />
Clock	28.322 MHz (from MSR 0x67)<br />
<br />
1. Understanding CR07 (Overflow Register, 0x07) = 0x1F (Binary: 00011111)<br />
The Overflow Register (CR07, Index 0x07) extends key vertical timing registers by providing their higher bits.<br />

Bit	Meaning	Your Value (Binary: 0001 1111)<br />
Bit 0	Vertical Total (CR06, bit 8)	1 (Extends 0xBF)<br />
Bit 1	Vertical Display End (CR12, bit 8)	1<br />
Bit 2	Vertical Sync Start (CR10, bit 8)	1<br />
Bit 3	Vertical Blank Start (CR15, bit 8)	1<br />
Bit 4	Line Compare (CR18, bit 8)	1<br />
Bits 5-7	Unused in standard VGA	000<br /><br />
2. Correcting Vertical Total with CR07 Overflow<br />
The Vertical Total Register (CR06, Index 0x06) = 0xBF (191 decimal) gets bit 8 extended from CR07.<br />
<br />
Formula to Compute Actual Vertical Total:<br />
Vertical Total=(CR07 bit 0×256)+CR06<br />
Vertical Total=(1×256)+191=447<br />
Final Vertical Total = 0x1BF = 447 scanlines.<br />
<br />
3. Re-evaluating Your VGA Resolution<br />
We now have:<br />

Horizontal Total (0x5F) → 768 pixels total width.<br />
Vertical Total (0x1BF = 447) → Similar to VGA 720x400 mode.<br />
Your hardware is running a standard VGA 720×400 text mode.<br />

Horizontal Timing (0x5F) → Matches VGA 9-dot font (720 pixels wide).<br />
Vertical Total (447) → Matches VGA text mode (400 visible lines + blanking).<br />
MSR (0x67) → 28.322 MHz clock, used in text mode.<br />
<h2>More detailed in Official Intel Documentation - SNB - Volume 3 Part 1: Display Registers - VGA Registers</h2>
https://www.intel.com/content/www/us/en/docs/graphics-for-linux/developer-reference/1-0/intel-core-processor-2011.html
<br /><br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Hello%20World%20Drivers/demo%205%20-%20VGA%20Controller%20WMD%20-%20related%20to%20EFI%20topic%20about%20VGA/69%20-%2008-02-2025%20-%20windows%20driver%20test%20for%20VGA%20controller%20registers.png?raw=true)

<br /><br />
References:<br />
[1] https://www.intel.com/content/dam/support/us/en/programmable/support-resources/fpga-wiki/asset03/basic-vga-controller-design-example.pdf <br />
[2] http://www.tinyvga.com/vga-timing <br />
[3] https://retrocomputing.stackexchange.com/questions/26243/how-do-80x25-characters-each-with-dimension-9x16-pixels-fit-on-a-vga-display-o <br />
[4] https://community.intel.com/t5/Embedded-Intel-Atom-Processors/Why-does-BayTrail-upscale-720x400-text-mode-to-800x600/td-p/204609?profile.language=en
<br /><br />
My comment to this? This is not straight forward what i'm thinking about it some time before :) But it is precisely by doing this kind of little research ( <b>IN MY FREE TIME</b> ), even if it is very cursory at the moment, that it allows me to get back on the track. To be a better programmer, more competent than just writing "hello world" code. But without those chips, hardwares etc, programmer doesn't exists... looking at it from the other side. I'm doing this for myself, not to be a pro programmer. I'm not aspire to that. As I wrote here before, but I will repeat. For this year 2025 I set myself a goal to make my own Virtual Machine. And as I thought at the beginning, it will allow me to better explore the details of the kernel, systems, hardware. And so it happens. 1 year. And I will see what happens in a year, what I will be able to do. examine VGA is the next step to the main goal.


