
> [!IMPORTANT]
> This is not "clean" version. I changed the pci files a bit https://github.com/tianocore/edk2/blob/master/ShellPkg/Library/UefiShellDebug1CommandsLib/Pci.c a few lines. So, similar to what you can see in the pictures here. In some time I will post "clean" versions from the base code. Now I did it quickly. This is only simple guide to setup serial communication in QEMU and compile OVMF project to use it.


This is for information only. I think I'll do some digging in QEMU in the coming weeks first. VirtualBox is licensed, qemu is more open source. Simple for learning.
<br /><br />

<h3>Build OVMF</h3>

DO NOT BUILD IN THIS WAY - SOURCE_DEBUG_ENABLE=TRUE - This didn't work for me. DO NOT USE SOURCE_DEBUG_ENABLE=TRUE ( on Windows only ? TODO )

```
# from edk2 root
make -C BaseTools
. edksetup.sh

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
. edksetup.sh
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

4. QEMU waits for PUTTY to open on port 4444<br />
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
