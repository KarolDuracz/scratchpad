As I review this topic I realize more and more that this is a long list of topics that make up the entire system. I'm talking about the bootloader and what happens at the beginning.
Even before the bootloader itself, there is the topic of the BIOS and UEFI itself. And the MBR and GPT partitions. This requires at least a sketch. Then there is the bootloader phase. And then the x86 system, which starts with ACPI, Timers, Interrupts, Watchdog, I/O ports etc.
For each topic I need a separate folder here. There is no detailed list of topics yet. But today I know that I need to systematize my knowledge on this subject in some way.
<br /><br />
And how threads works, how multiprocessor system works, how scheduler works on x86 Linux/Windows from the inside etc etc. All this things is on the Internet or in the manufacturer's documentation, but I want to systematize it here in this repo at least a bit.
<br /><br />
And finally something about USB, CAN, SERIAL, etc. And again, there are many cool materials and books about USB itself etc, but...
<br /><br />
I think this needs to be done at least as a sketch of how it works and looks before attempting to make a virtual machine and emulate devices. Because I set myself a "goal" to do it as an exercise. And in the win32 folder here is the beginning of a topic about it, but a few months later I know that it is a more complex topic and it cannot be done on 1 page "README.md", it must be divided into subtopics, stages, etc. This requires breaking down into stages and plans...
<br /><br />
But at the moment I am not interested in delving into the subject so much that to understand how to build my own CPU or architecture in detail at the electronics level. I am only interested in understanding the basics, to reach the next level in education. To use knowledge and these technology and devices in practice. Because we live in times when many people have done Reverse Enginering, even Intel 4004. Why do the same thing twice? Something else has to be done here...
<hr>
Update - 28-11-2024 - There is no way to learn all things based on history of development each devices and technologies in as short a time as one year 2025. And I mean not only computing systems themselves (microchips develop not as simple calulator to do basic maths, but complex machine to execute program. If we go back to the beginnings of this type of device) but also other things like hard drives, RAM, displays. There are tons of books for each of these topics. (...) Stopping for a moment only on displays. What the first monitors looked like, or displays in portable devices. What the code looked like, programming it, and how it looks today. Such a historical outline, even briefly, gives an insight into how technology has changed and what the beginnings of the "invention" looked like. And how it is done today. Because the display is a fundamental element of modern systems. One of the elements.
 <br /><br />
When I think today how to plan this all things to do it in 2025, for myself, to improve my programming skills, I see a sequence of certain topics that I would like to focus on. And first of all, it is x86 and ARM. Because these are proven architectures that are successfully used. In some time, other devices will probably be created to accelerate neural networks and this type of calculations, such as dedicated GPU systems on motherboards, next to the CPU.Even if current systems based on electrons, which need a lot of energy, are replaced by systems "powered" by lasers - photons, which do not have heat losses, and do not use as much energy, they will still be based on the binary system 0 and 1, on the standards that are used today, only maybe they will be much faster and will need less energy, only to operate lasers etc. But these will still be the same sets of instructions etc. at the beginning in my opinion. But this is just theorizing.<br /><br />
Starting with the market segment CPU:<br />
- x86 -> servers and personal computers<br />
---: A different board and build is for a laptop, a different one for a stationary PC, a different one for a server.<br />
------: <b></b>and here is the place when this repo starts and topics to cover from that moment on</b><br />
- ARM -> mobile devices<br />
---: what does the structure of today's phone (smartphone) look like?<br />
------: it will be maybe in 2026<br />
<br />
For now, this is only challange for myself to remember certain things and raise the current skill to the next level in some way. Public repo is a form of exhibitionism in some ways. But even though I show my flaws, imperfections, mistakes here, I still have a chance to confront other people and their views or opinions. And it gives me more discipline to do it systematically. I don't intend to put myself in the role of a mentor or scholar or to show how to do something, I'm just doing it for myself. That's it. 
 <br /><br />
 At the end of 2025 I want to know !!! (this is a general description) <br />
 [ 1 ] How the system starts and what happens after pressing POWER ON until Windows / Linux starts.
<br />
[ 2 ] How looks like internals of system, how window applications works from the system side etc
<br />
[ 3 ] How to build virtual machine x86 and how to emulate devices
<br />
[ 4 ] To improve quality and methods, to do that these all things
