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

for the paths environment I gave the location as NAME_PREFIX. Without nasm.exe in path, only target folder.

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

And then https://github.com/tianocore/tianocore.github.io/wiki/Windows-systems this guide. And instal nasm and setup NASM_PREFIX. And that's it. For win 8.1 right now I can't build this all things. I have many errors. But on the fresh installation of windows 10 with NASM 2.16.03 and MSVC2022 it just works just like that.
