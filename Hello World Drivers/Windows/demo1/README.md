![448 - 30-10-2024 - sciezki do waznych narzedzi](https://github.com/user-attachments/assets/39a21f18-6d52-48d8-a8f5-68dcfca17e9f)Simple Hello World
<br /><br />
Tested on Windows 8.1 64 bit.
<br /><br />
quick demo how to compile and how to get this message from driver. <br />
1. You must RESTART computer with hold SHIFT (left shift) to enter to restart boot option. Disable Device Driver Signing. <br />
https://learn.microsoft.com/en-us/windows-hardware/manufacture/desktop/boot-to-uefi-mode-or-legacy-bios-mode?view=windows-11 <br />
https://www.simple-shop.si/en/disable-enable-driver-signature-enforcement-on-windows-10
2. Without entering the system in that mode, we will not be able to load the driver as a services. We get message [SC] StartService FAILED 577. Windows cannon verify the digital signature....
3. (Probably I used 64 bit environment for CL.exe but...)

```
C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64
```

4. Run service - run CMD as Administrator and execute these commands<br />
<b>Create Service:</b> sc create HelloWorld binPath= "C:\DriverProjects\HelloWorld\HelloWorld.sys" type= kernel<br />
<b>Start Driver:</b> sc start HelloWorld<br />
<b>Stop Driver:</b> sc stop HelloWorld<br />
<b>Delete Driver:</b> sc delete HelloWorld<br />
5. When you have problem with stop or delete driver remove from REGEDIT.exe and restart computer
6. To catch message from driver I used "Dbgview.exe" (x86, not x64) as Administrator.
7. 

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Hello%20World%20Drivers/Windows/demo1/450%20-%2030-10-2024%20-%20c.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Hello%20World%20Drivers/Windows/demo1/448%20-%2030-10-2024%20-%20sciezki%20do%20waznych%20narzedzi-1.png?raw=true)
