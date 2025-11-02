> [!WARNING]
> If you want to test, do it in QEMU / VirtualBox. Here are images from my ASUS, but I do it at my own risk.

> [!IMPORTANT]
> Please read note at the bottom of the page


<h3>File description</h3>

```HelloWorld.c``` - https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/HelloWorld.c - source code

```HelloWorld.inf``` - https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/HelloWorld.inf - inf file for this demo

```bootx64.efi``` - https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/bootx64.efi - Compiled X64 demo shell with these patches from this source code that you put into /EFI/Boot/ - you can change name of file if you want.

```log-20251101-192622-0001.txt``` - https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/log-20251101-192622-0001.txt - log generated on real hw on usb flash driver to \EFI\Boot\myLogs\

<h3>How to use (example) - capmem command</h3>

```capmem start 5000 512``` — allocates 5,000 slots × 512 chars/line (capped by the safety limits).

```capmem status``` — see allocation & activity.

```capmem stop``` — stop writing new lines (you can still capmem start again to resume, or keep the buffer).

Run loadimg or other program — all OutputString/Print text will be copied into the memory slots as it arrives.

```capmem save``` — writes the captured lines to \EFI\Boot\myLogs\log-...txt.

```capmem free``` — free the capture buffer.

Added log collection mechanism from 12.4.3. EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL.OutputString() https://uefi.org/specs/UEFI/2.10/12_Protocols_Console_Support.html#efi-simple-text-output-protocol-outputstring - In short, it collects logs sent to OutputString from functions like Print and others that display text on the console using this protocol. This allows to launch applications with the "loadimg" command, and all logs generated from those applications will be in this buffer.
<br /><br />
Simply start with the command "capmem start 5000 512" . It can run in the background, and you can continue using this shell as I showed in demo #17. And these logs will be collected for this temporary buffer. We can then check the status with the command "capmem status". Sometimes you need to turn it off, e.g. when you want to use the editor to check the logs directly from this shell. Because it collects logs all the time and then it takes the whole buffer and suddenly it becomes 3000, 5000 lines and more when you scroll up and down the pages in the editor. That's why there is a stop order ```capmem stop``` . Then you can resume it again, e.g. by setting the same buffer parameters again: capmem start 5000 512.
<br /><br />

First entering the shell. And the "help" command. Basic description of commands and arguments for CAPMEM.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/images/1762021892014.jpg?raw=true)

Starting the buffer and activating capmem. You can also see "capmem status" command which shows information about whether it is active or how many lines are collected in the buffer.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/images/1762021892002.jpg?raw=true)

Then I will test load some demo with the LOADIMG command, here is a demo that shows the registers responsible for the resolution in Intel HD 3000 (I am not uploading this demo to github for now). Just make it print logs.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/images/1762021891989.jpg?raw=true)

Then another demo of some sort. This one is a GOP demo with custom fonts.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/images/1762021891976.jpg?raw=true)

And after exiting demo and type "help" again, let's assume it produced 105 lines in this case. However, subsequent demos will add additional lines from the logs, and this can be monitored with "capmem status."

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/images/1762021891963.jpg?raw=true)

<b>Let's assume that I want to save the log now on a USB flash drive to the path L"\\EFI\\Boot\\myLogs"</b> - ```capmem save``` - But from tests it turned out that for large logs, e.g. 3000-5000 lines, it can take up to 30-60 seconds.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/images/1762021891951.jpg?raw=true)

Ok, stop it. I want to enter the "editor.efi" demo to read this last log.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/images/1762021891926.jpg?raw=true)

Program is waiting for the index to be entered

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/images/1762021891913.jpg?raw=true)

19 is the index of this last log. Enter.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/images/1762021891901.jpg?raw=true)

In editor mode, you can see how many pages there are, you can scroll the buffer up/down with the PAGE UP/PAGE DOWN keys (without shift). Exit with the letter "q".

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/images/1762021891838.jpg?raw=true)

> [!NOTE]
> This file contains a lot of code that isn't compiled. Don't change anything at first, just compile source code as you download it. There's also a macro in line 28, but don't include it. Compile this code without changes now to get the exact same .efi file I posted here. It needs to be checked and rewritten. I'm public it in the same state as it was last time . https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/HelloWorld.c#L28 - because I had to debug a bit but I didn't want to lose code that was at the begining etc etc. I've added a macro, and it simply compiles whatever is correct in this code. So, when you read this code first time, remember that if there's an #else and the __FIRST_WORKING_VERSION__ macro is commented out, the second part of this compilation condition is executed. In short, compile as it currently is in the source code.

<h3>Running on Vritual Box - few commands example</h3>

Before entering this GOP demo (loadimg helloworld.efi), I turn on ```capmem start 5000 512``` so that I could save the log later. This is an example of it collecting logs all the time.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/images/virtual_bxox1.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/images/virtual_bxox2.png?raw=true)

it works the same on QEMU too.

<h3>Transparent effect demo</h3>

There is a list of parameters for the GOP command. These are further expanded, e.g. ```gop fontbg transparent``` disables background coloring under the font. ```fontfg``` is supposed to change the color of the font itself
<br /><br />
Virtual Box

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/images/virtual_box3.png?raw=true)

QEMU

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/images/qemu%20-%20cant%20detect%20mouse%20usb%20device.png?raw=true)

REAL HW

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/images/1762039321366.jpg?raw=true)

<h3>SIMPLE POINTER on real hw</h3>

the command ```gop pointer_mode device``` detects 2 handlers. You can see below that some kind of log appears, a few lines, and this is after pressing the left or right mouse button and moving it

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/images/1762039321340.jpg?raw=true)

```gop pointer_mode on``` turns on the pointer and draws a cursor on the screen. You can move it around by holding down a key and moving the mouse.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/images/1762039321290.jpg?raw=true)

Save the log (capmem was on )

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/images/1762039321277.jpg?raw=true)

link to log : https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/log-20251102-002033-0001.txt

<h3>Example of using capmem on real hw for a longer log from several demos</h3>

I used demo0 and demo4 from https://github.com/KarolDuracz/scratchpad/tree/main/bootloader_x86/tianocore%20EDK2/demo18%20-%20Summary%20of%20what%20I%20have%20now/part2 - there was supposed to be a demo1 but something went wrong when I quickly copied it. If you go back to demo #18 and look at the photos, they have exactly the same values ​​as you see in this log now.
<br /><br />
update 02-11-2025 - 11:00 - LOG from real hw : https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/log-20251102-104501-0001.txt
<br /><br />
short listing by line

```
[3] capmem: active=1 buffer_allocated=1 lines_stored=2 capacity=10000 chars_per_line=512
[5] gop set: mode changed to 0
[36] loadimg usb0.efi
[78] loadimg usb1.efi // It was supposed to be demo1 but it copied demo0
[120] loadimg usb4.efi
[830] on those lines you will see what I've uploaded to demo #18 in these noise images. The same logs
[916] capmem status
[918] capmem save
```

The file isn't large, less than 1000 lines, 72 KB. It took a ~10-15 seconds to save. But for sequences of such short diagnostic tests / demos it helps when the log starts getting longer. 
