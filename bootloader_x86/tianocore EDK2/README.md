<h2>Only information for myself</h2>

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/17%20-%202-11-2024%20-%20edk2.png?raw=true)

I tried to build on Windows 8.1 but so far no success. So as in the case of MSVC2022 I did it as a test on a Windows 10.0.19045.5131 virtual machine. As you can see in the picture.
<br /><br />
I used this video tutorial to get git clone, init and using nmake build. 
https://www.youtube.com/watch?v=jrY4oqgHV0o&ab_channel=papst5
But after that, I came back to Conf/target.txt and build version like on this tutorial https://www.youtube.com/watch?v=V2EAccnaSvo&ab_channel=AshrafAliS for 


```
TARGET    = EmulatorPkg/EmulatorPkg.dsc
TARGET_ARCH  = X64
TOOL_CHAIN_TAG  = VS2019
```

for the paths environment I gave the location as NASM_PREFIX. Without nasm.exe in path, only target folder. 

```
C:\Users\test\AppData\Local\bin\NASM\
```

And like on this tutorial https://www.youtube.com/watch?v=V2EAccnaSvo&ab_channel=AshrafAliS I ran WinHost.exe. And it works. So I can analyze a little bit how it works meantime. In this "AshrafAliS" channel, there is playlist how to write EFI demos.
<br />
<hr >
<br />
All guide is here https://github.com/tianocore/tianocore.github.io/wiki/Windows-systems but these 2 videos is great. For Windows 10 https://github.com/tianocore/edk2 just download git and initialize like here

```
git clone https://github.com/tianocore/edk2.git
cd edk2
git submodule update --init
cd ..
```

```
// on the page https://github.com/tianocore/edk2 look on the right side - Releases 29
// edk2-stable202411
// Latest
// last week
// + 28 releases
// I used this command before nmake from this tutorial https://www.youtube.com/watch?v=jrY4oqgHV0o&ab_channel=papst5
git tag
git checkout tags/{copy here last version} // current https://github.com/tianocore/edk2/releases/tag/edk2-stable202411
```

And then https://github.com/tianocore/tianocore.github.io/wiki/Windows-systems this guide. And instal nasm and setup NASM_PREFIX. And that's it. Then I used guide from papst5 channel.

```
cd BaseTools
nmake
```

And then back to AshrafAliS's version, and build for X64, emulator, to get WinHost.exe
 <br /><br />
For Win 8.1 (my current host OS which I use) right now I can't build this all things. I have many errors. But on the fresh installation of windows 10 with NASM 2.16.03 and MSVC2022 it just works just like that.

<hr>
This gives me a way to explore the bootloader more. This includes GRUB and Legacy BIOS at some point once I get through the basics of EDK.
<br /><br />
https://tianocore-docs.github.io/edk2-BuildSpecification/release-1.28/#edk-ii-build-specification <br />
https://tianocore-docs.github.io/edk2-BuildSpecification/release-1.28/4_edk_ii_build_process_overview/#4-edk-ii-build-process-overview
<hr >
Beside tianocore.But for now more interesting me deeply understand windows bootloader. But Linux is more straightforward in its approach to many things. <br />
https://www.gnu.org/software/grub/manual/grub/grub.html <br />
https://github.com/rhboot/grub2
<hr>
But looking deeply into guidelines https://github.com/tianocore/tianocore.github.io/wiki/start-using-UEFI. For example here 
https://github.com/tianocore/tianocore.github.io/wiki/EDK-II-Platforms they write, to build environment for Windows I need https://github.com/tianocore/tianocore.github.io/wiki/Nt32Pkg . TODO  <br />
And then back to this https://github.com/tianocore/tianocore.github.io/wiki/Windows-systems#build as it is written at the bottom 
<br /><br />

```
As a tangible result of the build, you should have the HelloWorld UEFI application. If you have a UEFI system available to you which matches the processor architecture that you built, then this application should be able to run successfully under the shell.
<br />
C:\edk2> dir /s Build\MdeModule\DEBUG_...\IA32\HelloWorld.efi
```


![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/19%20-%202-11-2024%20-%20edk2%20test.png?raw=true)

<br />
but this needs analysis etc. to write something more sensible on this topic. Second case. In this efi demo list is UiApp.efi. And for now, what interests me most is how it is built this UiApp.efi. TODO.

<br /><br />
<hr>
btw. This YT channel is very helpful, this playlist about EDK2 https://www.youtube.com/watch?v=1qdk0XJ6gCw&list=PLz-YdBAdSGeh8MT3c7_sfolRzXM8cNXPk&ab_channel=AshrafAliS to see how it is done in practice, specifically. 
<hr>
In other hand. What I found. FreeBSD have a nice handbooks and tutorials.  There is also a topic about booting the kernel
https://man.freebsd.org/cgi/man.cgi?query=uefi&sektion=8&apropos=0&manpath=FreeBSD+14.1-RELEASE+and+Ports <br />
https://man.freebsd.org/cgi/man.cgi?query=efibootmgr&sektion=8&apropos=0&manpath=FreeBSD+14.1-RELEASE+and+Ports <br />
https://docs.freebsd.org/en/books/ <br />
And this book -> FreeBSD Architecture Handbook
https://docs.freebsd.org/en/books/arch-handbook/boot/
From what I see, for many topics it is better to use Linux than to look for help in Windows topics to better understand these topics from the Windows side...

<hr>
And last words in this introduce. I wonder more and more why I'm doing this, why the hell do I need it  :sweat_smile: . This is hard. But I think it's the best way to touch on the subject of hardware in the future. I mean the motherboard, and the devices connected to it. And somehow to remind the basics. To also better understand the IT environment in general. And to have a chance and opportunity to build something in future maybe...
<hr>
I forget about python. I installed on Win10 latest version https://www.python.org/ftp/python/3.13.0/python-3.13.0-amd64.exe - probaly I set first in CMD this variable usign command like that. And this was probably before run edksetup.bat.
<br />

```
set PYTHON_HOME=C:\path\to\python
echo %PYTHON_HOME% // if exists return path to app python
```

And you can then move the "Build" folder with the compiled software to Win 8.1 (for my case) and it works. btw. Virtual Box is slow. Better is to install winrar or something like that and compress and then move this compressed file between guest <> host  

<hr>
Update 3-12-2024 - I send few "prompts" (questions) to free version 4o ChatGPT (<i>on EDK2 to build custom bootloader OVMF, devices tree, power management, multicore support etc etc - I have answer how to build simple custom OVMF and kernel to run without qemu or virtual box and under these virtual machines. But this is only text from prompt. I don't check this at this moment. But even this small tiny example shows me, that it is better to create a guide with ChatGPT</i>). This system is very good for many case. This deep neural network undestand raw binary code from firmware (random binary bios .bin downloaded from internet) in some way. It uderstand machine code just like that and it can break it down into stages to explain what is that and where does it come from probably (<i>this "feature" and deeply understand .bin file can have serious applications in terms of security, but for example CPU execute each instruction and when something goes wrong it sends a signal exception. But this is like stereo mixer or something like that, and you put raw binaries and you get outputs for which the model was trained for. The interesting thing is that this model understands random BIOS binary code in my case for ASUS. People even have trouble understanding machine code, which is why there are assembler mnemonics. And high-level languages ​​like C. This is very interesting. But just as interesting is the history of the creation of this GPT model from an ordinary perceptron, where it was once considered useless because, according to the first reports on this subject, it could not solve the XOR operation. And changes between 2010-2017, including the creation and development of AlexNet, Transformer. In comparison, just like the creation of the Colt pistol, then machines in World War II, all the way to the atomic bomb. These were moments that shaped reality. In the same way, these models shape our reality and future today. Sorry for philosophizing, the history of inventions that changed the world is interesting. Anyway... </i>). This system makes mistakes, sometimes misleads, but generally suggests accurate solutions. And I think the best choice will be to create a guide at the end, basing it among other things on answers to given topics from Chat GPT. Because why not. Model 4o is competent enough that by checking and correcting some errors I can create a really interesting tutorial for this complex topic. And also give space to further develop, review, analyze this topic further in the future. Because there are many things that need to be broken down into stages, into smaller pieces. And throwing such junk code as in this "scratchpad" in pieces, taken out of context, makes no sense without a thorough explanation, i.e. a simple text on 2-3 pages, possibly a longer one as an introduction or detailed discussion. I know that, but again, this scratchpad in current way is to start somwhere diging these topics. I don't feel competent enough to describe it in my own words at this point, and I doubt it will be any different in a year. It's too short a time for that. But based on GPT 4o it could be valuable. <br /><br />
because I'm not making my CV or anything like that here to show what I know, what I learn, but to explore these topics as best as possible. And for it to really have value, for it to be a point of reference. This is the main goal.
<hr>
And one more thing. Since I started messing around with WinPE, I've been thinking more and more about packing it up later as a WinPE image, I mean installing basic tools, setting up the environment and building an ISO. I don't know if it's legal to do things like that and then share them. But I'm thinking more and more about this solution. Downloading an ISO as a LiveCD to run on a virtual machine without installing it is easier than compiling, building, setting up the environment. It's better to do it all once. I'll just think about it again, whether it's legal and whether I can do it. And whether the standard 72h of kernel operation doesn't interfere, etc.
What I mean is just installing everything that's necessary for this demos, to work and just running it as a process, just like that, as LIVE CD WinPE. I'll think about it...
<hr>
<h2>TODO in 2025 for achieve goal - to build own virtual machine and minimalist custom OS for educational purpose.</h2>
This repo exists among other things, to understand how it works. And how OS handle this after boot stages. This is complex topic, but I want to understand. There are many topics around this. <br /> <br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/how%20it%20works%20-%20achieve%20in%202025.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/load%20kernel%20bootx64%20efi%20from%20cdrom.png?raw=true)
