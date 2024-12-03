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
