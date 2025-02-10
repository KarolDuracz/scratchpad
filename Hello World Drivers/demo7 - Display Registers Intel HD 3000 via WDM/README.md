<h2>TODO</h2>

https://www.intel.com/content/www/us/en/docs/graphics-for-linux/developer-reference/1-0/intel-core-processor-2011.html
<br /><br />
Two most important datasheet for this demo 7: <br />
SNB - Volume 1 Part 1: Graphics Core - Offsets for Pipe A and Pipe B to MMIO <br />
SNB - Volume 3 Part 2: Display Registers - CPU Registers - full configuration and description registers like PIPEBCONF etc. Page 7 - 10 Display Mode Set Sequence. This is full guideline step by step.
<br /><br />	
At this moment I only checked registers for PIPE A and PIPE B, and do simple test to write directly through PIPE_SRCDIM_A to set resolution 
from 1600x900 to 1420x900. Because I got all the values ​​from 1024x768 - 1600x900 which I can set to DISPLAY A. And this really set resolution but display in Windows is cut off to the left, and does not adjust icons, bottom bar to the current resolution. Also the control panel and resolution change options are not updated. It is the task of the operating system to change these values ​​in the system GUI.
<br /><br />	
It works and that's what matters most to me at the moment. But configuring both pipelines is complex and requires reading the documentation to attempt on UEFI.
<br /><br />	
No details yet. It is important that in my case Intel HD 3000 has such addresses in Device Manager (Resources tab): <br />

```
Memory Range : 00000000DD400000 - 00000000DD7FFFFF
Memory Range : 00000000B0000000 - 00000000BFFFFFFF
I/O Range : E000 - E03F
----- irq ---
0xFFFFFFFE (-2)
----- VGA ---
I/O Range : 03B0 - 03BB
I/O Range : 03C0 - 03DF
Memory Range : 00000000000A0000 - 00000000000BFFFF
```

That's why I use for this DD400000 address directly for MMIO. For UEFI this is BAR0-6 range in PCI header.
https://github.com/KarolDuracz/scratchpad/blob/main/Hello%20World%20Drivers/demo7%20-%20Display%20Registers%20Intel%20HD%203000%20via%20WDM/PrivilegedInstructionsDriver.c#L119C26-L119C38
In line 201 i 202 I just tried to enter new values ​​for other resolutions read earlier and it works. But it is only a change of resolution. I need more information about improving the configuration.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Hello%20World%20Drivers/demo7%20-%20Display%20Registers%20Intel%20HD%203000%20via%20WDM/99%20-%2010-02-2025%20-%20ok%20mam%20oba%20pipeline%20do%20monitorow%20ale%20to%20jest%20bardziej%20zlozona%20konfiguracja.png?raw=true)

<h2>TODO</h2>
