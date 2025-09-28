<h2>Demo 11 - Install ACPI table protocol in EmulatorPkg</h2>

In the previous demo (demo 10 https://github.com/KarolDuracz/scratchpad/tree/main/bootloader_x86/tianocore%20EDK2/demo10%20-%20hello%20world%20extended), I linked to an 
interesting repo with many lessons (https://github.com/Kostr/UEFI-Lessons/tree/master). Someone did a good job. Even in the days of GPT Chat, you need documentation and relevant data to ask the right questions that 
will give you a solution or idea.
<br /><br />
I've looked at a few that could easily be implemented on "HELLO WORLD EXTENDED" from demo10 in EmulatorPkg, but one is interesting. I've been wanting to figure this out 
for a few months now. It's about "Lesson 37: Investigate ways how to add acpiview command functionality to your shell" https://github.com/Kostr/UEFI-Lessons/tree/master/Lessons/Lesson_37
<br /><br />
By default, ACPI VIEW APP is in the EDK2 folder in the shell module but by default it is not included in the EmulatorPkg project and compiled https://github.com/tianocore/edk2/tree/master/ShellPkg/Application/AcpiViewApp

<h3>Demo</h3>

TIP: The GOP in the EmulatorPkg base version has a limited text buffer, but you can scroll up and down a bit by holding SHIFT + PageUp / SHIFT + PageDown

<h2>Step 1</h2>

As with (Kostr-UEFI-Lessons), I first had to build this module and copy it to the folder where the shell sees it after entering FS0:

```
>build -p ShellPkg/ShellPkg.dsc -m ShellPkg/Application/AcpiViewApp/AcpiViewApp.inf
```

AcpiViewApp.efi is then copied to the other demo folder, where helloworld.efi is also located in the built EmulatorPkg. In my case, after building, I have it in Shell\DEBUG_VS2019\... and then I move it to EmulatorPkg where Winhost.exe is

```
//FROM
C:\Users\kdhome\Documents\progs\edk2_win81\edk2\Build\Shell\DEBUG_VS2019\X64\ShellPkg\Application\AcpiViewApp\AcpiViewApp\DEBUG

// THIS
C:\Users\kdhome\Documents\progs\edk2_win81\edk2\Build\EmulatorX64\DEBUG_VS2019\X64
```

After running "acpiviewapp" in the shell, I no longer see "'acpiview' is not recognized as an internal or external command, operable program, or script file." because I have already compiled it and copied it to the folder, so the shell sees the application. But it doesn't see the ACPI TABLE

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo11%20-%20Install%20ACPI%20table%20protocol%20in%20EmulatorPkg/step%201.png?raw=true)

<h2>Step 2</h2>

I run the "dmem" command. In case of OVMF there are some addresses here, for EmulatorPkg the addresses for ACPI Table are NULL.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo11%20-%20Install%20ACPI%20table%20protocol%20in%20EmulatorPkg/step%202.png?raw=true)

<h2>Step 3</h2>

I'm running helloworld.efi

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo11%20-%20Install%20ACPI%20table%20protocol%20in%20EmulatorPkg/step%203.png?raw=true)

<h2>Step 4</h2>

I check these tables again with the "dmem" command. Address is no longer NULL.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo11%20-%20Install%20ACPI%20table%20protocol%20in%20EmulatorPkg/step%204.png?raw=true)

<h2>Step 5</h2>

I run ACPIVIEWAPP again and as you can see it reads something

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo11%20-%20Install%20ACPI%20table%20protocol%20in%20EmulatorPkg/step%205.png?raw=true)


<h3>Short explanation</h3>

1. HelloWorld.c ( source code ) - The entire code - https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo11%20-%20Install%20ACPI%20table%20protocol%20in%20EmulatorPkg/HelloWorld.c
2. INF file - Compared to Demo10, I added another protocol and DebugLib to have logs in CMD on the right side during boot, not only in the GOP shell - https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo11%20-%20Install%20ACPI%20table%20protocol%20in%20EmulatorPkg/HelloWorld.inf
3. The compiled version from this demo is what I used here in the HelloWorld.efi image.- https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo11%20-%20Install%20ACPI%20table%20protocol%20in%20EmulatorPkg/HelloWorld.efi
4. Compiled acpiviewapp.efi (debug X64)- https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo11%20-%20Install%20ACPI%20table%20protocol%20in%20EmulatorPkg/AcpiViewApp.efi

