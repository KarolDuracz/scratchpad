<h2>Demo2</h2>
Execute unprivileged instruction to read MSR and CR0 state. 
<br /><br />


![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Hello%20World%20Drivers/demo2/17112024%20-%20pic1%20-%20read%20msr%20and%20cr0.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Hello%20World%20Drivers/demo2/17112024%20-%20pic2%20-%20EFER%20status.png?raw=true)

<h2>How to run</h2>
1. Follow this instruction how to reboot (https://github.com/KarolDuracz/scratchpad/tree/main/Hello%20World%20Drivers/Windows/demo1) - disable-enable-driver-signature-enforcement-on-windows<br />
2. Execute "run.bat" script. This compile all things (only without readmsr.asm - this is not needed here) - I use here __readcr0 (https://learn.microsoft.com/en-us/cpp/intrinsics/readcr0?view=msvc-170) and  __readmsr (https://learn.microsoft.com/en-us/cpp/intrinsics/readmsr?view=msvc-170) <br />
3. Create service (cmd as administrator)

```
sc create PrivInstDriver binPath= "C:\Users\kdhome\Documents\progs\__trash-22-10-2024-startfrom\17-11-2024\PrivilegedInstructionsDriver.sys" type= kernel
```
4. Open Dbgview (not dbgview64) - SysinternalsSuite tools (https://learn.microsoft.com/en-us/sysinternals/downloads/sysinternals-suite)
5. start service PrivInstDriver

```
sc start PrivInstDriver
```

6. sc stop PrivInstDriver - to stop
7. sc delete PrivInstDriver - to delete
8. If you want to run again -> open run.bat > then create service "sc create..." > then watch DbgView > start service ... 

<hr>
In "privileged.asm" you can execute a sequence of asm code and do an "insert" into the main code as shown by the CLI operation. You just have to do RET because there is a CALL before. But on this picture I do test to check EFLAGS (https://en.wikipedia.org/wiki/FLAGS_register) using (https://learn.microsoft.com/en-us/cpp/intrinsics/readeflags?view=msvc-170). This is only introduce... how to get access to these instructions. But EFLAGS in this demo don't show IOPL bit. Only first 10 bit max, for first test 0x282 and for second after CLI 0x82. This is only bytes 0-9.<br /><br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Hello%20World%20Drivers/demo2/17112024%20-%20pic3%20-%20eflags.png?raw=true)


