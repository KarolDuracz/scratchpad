```
update history
: #1 - 12-11-2025 - first post
: #2 - 13-11-2025 - add debug cygwin info
: #3 - TODO - add information to PcdLib.c(95) message, info chardev, gdb etc .
```


> [!NOTE]
> This is not "clean" version. I changed the pci files a bit https://github.com/tianocore/edk2/blob/master/ShellPkg/Library/UefiShellDebug1CommandsLib/Pci.c a few lines. So, similar to what you can see in the pictures here. In some time I will post "clean" versions from the base code. Now I did it quickly. This is only simple guide to setup serial communication in QEMU and compile OVMF project to use it.


This is for information only. I think I'll do some digging in QEMU in the coming weeks first. VirtualBox is licensed, qemu is more open source. Simple for learning.
<br /><br />

<h3>Build OVMF</h3>

DO NOT BUILD IN THIS WAY - SOURCE_DEBUG_ENABLE=TRUE - This didn't work for me. DO NOT USE SOURCE_DEBUG_ENABLE=TRUE ( on Windows only ? TODO )

```
# from edk2 root
make -C BaseTools
. edksetup.sh / edksetup.bat

# produce a DEBUG OVMF fd that sends debug to serial
OvmfPkg/build.sh -a X64 -b DEBUG -n $(nproc) -D DEBUG_ON_SERIAL_PORT -D SOURCE_DEBUG_ENABLE
# or, using edk2 build directly:
build -p OvmfPkg/OvmfPkgX64.dsc -a X64 -b DEBUG -D DEBUG_ON_SERIAL_PORT=TRUE -D SOURCE_DEBUG_ENABLE=TRUE
```

Ok, I saw the message from ASSERT, so it showed that communication on the serial port was working, wireshark also saw the messages, but QEMU was stuck on it and could not proceed. That's why there is a link at the bottom to a forum where someone had a similar problem.

```
SecCoreStartupWithStack(0xFFFCC000, 0x820000) ASSERT [SecMain] ...\MdePkg\Library\BasePcdLibNull\PcdLib.c(95): ((BOOLEAN)(0==1))
```
<h2>THIS IS CORRECT COMMAND FOR WINDOWS - this works for me</h2>

And files ```OVMF_CODE.fd``` and ```OVMF_VARS.fd``` that I put here are compiled from these commands

```
. edksetup.bat
build -p OvmfPkg/OvmfPkgX64.dsc -a X64 -b DEBUG \
  -D DEBUG_ON_SERIAL_PORT=TRUE \
  -D SOURCE_DEBUG_ENABLE=FALSE
```

<h3>Command to run qemu</h3>

1. ```OVMF_CODE.fd``` and ```OVMF_VARS.fd``` to Copy C:\Program Files\qemu ( main qemu folder )
2. Run as Administrator CMD ( for my case it is required ).
3. Run via this command ( nongraphic ) - you see on CMD on picture the same command

```
qemu-system-x86_64.exe ^
 -machine q35 -m 512 ^
 -drive if=pflash,format=raw,unit=0,file=OVMF_CODE.fd,readonly=on ^
 -drive if=pflash,format=raw,unit=1,file=OVMF_VARS.fd ^
 -serial tcp:127.0.0.1:4444,server,wait ^
 -debugcon tcp:127.0.0.1:4445,server,wait ^
 -global isa-debugcon.iobase=0x402 ^
 -nographic
```

4. QEMU waits for PUTTY to open on port 4444 - ( <b>look at the picture below, there is a picture of how I run putty</b> )<br />

```
Then open PuTTY and connect to 127.0.0.1:4444 (connection type: Raw) to see the guest serial console in real time. The file ovmf-debugcon.log will contain the very early DEBUG() output routed via IO 0x402.
```

5. Then second instance , run second PUTTY on port 4445 ( you see this on picture - on the right is 4444 for logs for serial communication )<br />
6. After going through all the logs it should enter SHELL. And as you can see, it responds to commands such as ```pci``` on the second picture.<br />


![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo20%20-%20QEMU%20serial%20port%20and%20messages%20in%20early%20booting%20phase%20and%20OVMF%20compilation%20guide/168%20-%2012-11-2025%20-%20DZIALA.png?raw=true)


I started modifying some parts of CORE and BUS, that's why there are these strange messages. It doesn't matter. It's just about how it starts

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo20%20-%20QEMU%20serial%20port%20and%20messages%20in%20early%20booting%20phase%20and%20OVMF%20compilation%20guide/170%20-%2012-11-2025%20-%20dziala%20-%20czyli%20zmiana%20PCI%20kodu%20-%20dziala.png?raw=true)

<h3>Test connection on Windows</h3>

Also wireshark on loopback

```
Test-NetConnection -ComputerName 127.0.0.1 -Port 4444
Test-NetConnection -ComputerName 127.0.0.1 -Port 4445
netstat -ano | findstr ":4444"
netstat -ano | findstr ":4445"
```


<h3>Links</h3>
1. https://github.com/tianocore/edk2/tree/master/OvmfPkg <br />
2. https://edk2.groups.io/g/discuss/topic/edk2_stable202205_issue/92898495?utm_source=chatgpt.com

<h2>DEBUG via GDB ( cygwin ) on Windows</h2>

I used this program to access GDB on Windows - https://www.cygwin.com/ . Below in the pictures is the version I have installed (uname -a).
<br /><br />
Command with additional parameter ``` -gdb tcp::1234 -S``` 

```
qemu-system-x86_64.exe ^
 -machine q35 -m 4096 ^
 -drive if=pflash,format=raw,unit=0,file=OVMF_CODE.fd,readonly=on ^
 -drive if=pflash,format=raw,unit=1,file=OVMF_VARS.fd ^
 -serial tcp:127.0.0.1:4444,server,wait ^   # guest console (PuTTY)
 -debugcon tcp:127.0.0.1:4445,server,wait ^ # early DEBUG() (isa-debugcon)
 -gdb tcp::1234 -S ^                         # open GDB server on host port 1234 and *do not run* until debugger attaches
 -nographic
```

QEMU has a command ```info chardev``` . This is sample information. To check if connections are established at an early stage.

```
What your info chardev means
debugcon: filename=tcp:127.0.0.1:4445,server=on <-> 127.0.0.1:62393
parallel0: filename=null
compat_monitor0: filename=stdio
serial0: filename=disconnected:tcp:127.0.0.1:4444,server=on


debugcon: ... <-> 127.0.0.1:62393 — QEMU has an accepted connection on the debugcon TCP listener (port 4445). a client connected from local port 62393. That means early EDK2 DEBUG() output is already being routed to a client (something connected to 4445).

serial0: filename=disconnected:tcp:127.0.0.1:4444,server=on — QEMU is listening on TCP port 4444 for the guest serial, but no client is connected to it right now. That’s why PuTTY shows nothing — PuTTY either didn’t connect or did not connect correctly to 127.0.0.1:4444.

compat_monitor0: filename=stdio — the QEMU monitor is attached to the terminal you started QEMU from (that’s why you saw the monitor).
Conclusion: early debug may be arriving on the debugcon port (4445) but your guest serial (4444) is still disconnected.
```

<br /><br />
I don't have much experience or knowledge about the internals of WinDbg and how the Windows kernel is debugged, but most guides point to settings via bcdedit, as in this example

```
bcdedit /debug on
bcdedit /dbgsettings serial debugport:1 baudrate:115200
```

THAT'S WHY WINDBG WAS "DISQUALIFIED". I had to find access to GDB that would see this connection, have access to this process via the port and an argument for qemu that opens communication for GDB. so I used ```Cygwin64 Terminal``` 
<br /><br />

In this image, you can see that:<br />
1. QEMU has an additional command ```-gdb tcp::1234 -S ^ ``` <br />
2. It's waiting for the first (of two) putty to open on 4444.<br />
3. I ran GDB with the command ```target remote :1234``` to check. After a few seconds, it returns a message about the connection timed out.<br />
4. At the bottom of the second window, you can see the cygwin version.<br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo20%20-%20QEMU%20serial%20port%20and%20messages%20in%20early%20booting%20phase%20and%20OVMF%20compilation%20guide/debug%20cygwin/debug%20gdb%20cygwin%201.png?raw=true)

Start first putty on 4444 port. At the moment there is still no connection because it needs a second putty on 4445.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo20%20-%20QEMU%20serial%20port%20and%20messages%20in%20early%20booting%20phase%20and%20OVMF%20compilation%20guide/debug%20cygwin/debug%202%20-%20wait%20for%20second%20putty.png?raw=true)

Turning on the second putty <b>In the meantime, I've run GDB to listen. It has some time to do this, so I quickly fire up Putty.</b>

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo20%20-%20QEMU%20serial%20port%20and%20messages%20in%20early%20booting%20phase%20and%20OVMF%20compilation%20guide/debug%20cygwin/3.png?raw=true)

Ok, it started and GDB caught it

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo20%20-%20QEMU%20serial%20port%20and%20messages%20in%20early%20booting%20phase%20and%20OVMF%20compilation%20guide/debug%20cygwin/4.png?raw=true)

First, checking the first state of the registers, theoretically this is the state after the CPU RESET, i.e. CPU initialization. ```i r ``` command.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo20%20-%20QEMU%20serial%20port%20and%20messages%20in%20early%20booting%20phase%20and%20OVMF%20compilation%20guide/debug%20cygwin/5.png?raw=true)

Using the ``si``` (step into) command, which is one instruction forward, I get this register state. It looks like an early entry to RESET VECTOR and the first instruction ``mov eax, cr0``` 

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo20%20-%20QEMU%20serial%20port%20and%20messages%20in%20early%20booting%20phase%20and%20OVMF%20compilation%20guide/debug%20cygwin/6.png?raw=true)

Ok, release dbg, continue executing the program. Then stop ```ctrl + c```. You can see the logs from loading the ovmf firmware in pytty.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo20%20-%20QEMU%20serial%20port%20and%20messages%20in%20early%20booting%20phase%20and%20OVMF%20compilation%20guide/debug%20cygwin/7.png?raw=true)

I'll go back to what I did at the beginning for a moment. Before starting, I checked the connection. Does Cygwin see these connections? As you can see, it does.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo20%20-%20QEMU%20serial%20port%20and%20messages%20in%20early%20booting%20phase%20and%20OVMF%20compilation%20guide/debug%20cygwin/8%20netsat.png?raw=true)

Ok, continue to enter shell

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo20%20-%20QEMU%20serial%20port%20and%20messages%20in%20early%20booting%20phase%20and%20OVMF%20compilation%20guide/debug%20cygwin/9%20-%20continue.png?raw=true)
