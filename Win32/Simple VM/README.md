<b>Simple Virtual Machine</b><br />
1. Simulation console I/O, interrupts and run code from file .name as we see on editor.<br />
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple%20VM/output_vm_console_test.gif?raw=true)
<br />2. Print "hello world" text on the console. Standard hello world demo. <br />
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple%20VM/output_vm_hello_world.gif?raw=true)
<br />3. "My" simple implementation of this but a lot of stuff TODO. But this is not important here right now. Explanation is above this pictures. <br />
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple%20VM/output_simple_vm_win32.gif?raw=true)
<br /><br />
When you start learning low level programming, you wonder how this code executes etc. And at some point you learn the C language, then the gcc compiler, then you look into the assembler code. And then you come across NASM. Then you also go deeper, learning the processor's Instruction Sets (https://en.wikipedia.org/wiki/X86_instruction_listings) etc. Maybe you want to find how this works on the chip side, on the silicon like this guy (https://www.youtube.com/watch?v=IS5ycm7VfXg&ab_channel=SamZeloof) - "Z2" - Upgraded Homemade Silicon Chips
<br /><br />
ANYWAY...
<br /><br />
The first two demos show an example of a VM implementation from the book "Understanding Programming - Gynvael Coldwind" (https://ksiegarnia.pwn.pl/Zrozumiec-programowanie,216633888,p.html?wnpwn&cq_src=google_ads&cq_cmp=8482839064&cq_term=&cq_plac=&cq_net=g&cq_plt=gp&cq_src=google_ads&cq_cmp=8482839064&cq_term=&cq_plac=&cq_net=g&cq_plt=gp&gad_source=1&gclid=EAIaIQobChMIw9qJtcvCiAMV_IxoCR2cGABOEAQYASABEgJzBfD_BwE) . And this is more complex example that my simple trash code. Gynvel's example has a code compiler using NASM as seen in the first console demo. It has interrupt handling, I/O events. Instructions have their size in bytes for IP (instruction pointer) etc. This is very useful demo to learn basics about "how it works".
<br /><br />
But today it's just fun for yourself because this problem has long been addressed and solved at the root. Thanks to this, today the client can connect to his virtual machine and use its resources. Today it's not such an interesting topic for me, but I want to have it here for personal reasons, to just get it out of my head. For there to be something on this topic, to have a point of reference. Today, what seems more interesting to me is the operation of virtual servers and machines to which many clients can connect... but still, for yourself, you need to know how to implement even a simple VM to go deeper into this topic, this issue.
<br /><br />
link to other x86 emulators and other interesting emulators:
<br />
These two are interesting because they can put a ready Linux system in a minimal kernel version. But I'm not sure because I haven't tested it thoroughly. But this is already a much more complex example of a VM / emulator than my demos and gynvel's.
<br />
https://github.com/copy/v86 ==> https://copy.sh/v86/ -- x86 emulator can boot linux I guess
https://github.com/jart/blink -- x86 emulator can boot linux I guess
<br />
https://github.com/redcode/6502
https://github.com/redcode/Z80
<br /><br />
nasm which I used here to compile this in the pictures: <br /> 
NASM version 2.16.03 compiled on Apr 17 2024
https://www.nasm.us/pub/nasm/releasebuilds/2.16.03/win64/
https://www.nasm.us/
<br /><br />
Today when you talk about VM the first thing that comes to mind is qemu or Virtual Box etc... But this is a complex machine. As a beginner, you have to start somewhere...
<br /><br />
 - - - - - 
 <br />
 In scratchpad/Win32/Simple VM/simple_virtual_machine/simple_virtual_machine/simple_virtual_machine.cpp
<b>there are several implementations of my VM commented out. The 3 in this image are from one of the implementations in the .cpp file. The last compiled code can execute simple MOV, ADD, SUB instructions as you see on the picture. On the right side write this pseudo asm code, and then click RUN. And PAUSE or STOP.<b/>
 <br />
![dump](https://raw.githubusercontent.com/KarolDuracz/scratchpad/main/Win32/Simple%20VM/54%20-%2014-09-2024%20-%20vm%20cd.png)
