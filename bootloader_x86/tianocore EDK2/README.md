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
