Tested on Windows 8.1 64 bit.
<br /><br />
quick demo how to compile and how to get this message from driver. <br />
1. You must RESTART computer with hold SHIFT (left shift) to enter to restart boot options. Disable Device Driver Signing. <br />
https://learn.microsoft.com/en-us/windows-hardware/manufacture/desktop/boot-to-uefi-mode-or-legacy-bios-mode?view=windows-11 <br />
https://www.simple-shop.si/en/disable-enable-driver-signature-enforcement-on-windows-10
2. Without entering the system in that mode, we will not be able to load the driver as a services. We get message [SC] StartService FAILED 577. Windows cannot verify the digital signature....
3. (Probably I used 64 bit environment for CL.exe but... the same as Windows system) 

```
C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64
```
Compile using script
```
run.bat
```

4. Run service - run CMD.exe as Administrator and execute these commands<br />
<b>Create Service:</b> sc create HelloWorld binPath= "C:\DriverProjects\HelloWorld\HelloWorld.sys" type= kernel<br />
<b>Start Driver:</b> sc start HelloWorld<br />
<b>Stop Driver:</b> sc stop HelloWorld<br />
<b>Delete Driver:</b> sc delete HelloWorld<br />
5. When you have problem with stop or delete driver remove from REGEDIT.exe and restart computer (with Disable Device Driver Signing)
6. To catch message from driver I used "Dbgview.exe" (x86, not x64) as Administrator.


![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Hello%20World%20Drivers/Windows/demo1/450%20-%2030-10-2024%20-%20c.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Hello%20World%20Drivers/Windows/demo1/448%20-%2030-10-2024%20-%20sciezki%20do%20waznych%20narzedzi-1.png?raw=true)

<hr>
<br />
This is simple demo to start somewhere. At this moment I have few demos, but I haven't tested them yet. These demos create a driver and then from the user space it communicates with the driver. But for now I need to get to know this mechanism better. So far I have managed to successfully compile a simple demo and receive the message from the driver in some way.
<br /><br />
There are a few topics behind this that I have learned about along the way, but I am not posting anything more about it here at the moment.:<br />
1. The first thing to achieve is to configure the environment and compile and link. There are a few mistakes that are worth mentioning by the way related to linking headers and ntoskrnl.lib. And with /ENTRY:DriverEntry /SUBSYSTEM:NATIVE.<br />
2. using SC.exe to create and manage service (driver)<br />
3. using  makecert.exe  , signtool.exe  and   certmgr.msc  to create self signed dertificate<br />
4. enable kernel debugging and debug it with kd.exe - bcdedit /debug on<br />
5. How to configure windbg to catch this message - how to capture the KdPrint messages<br />
etc.
