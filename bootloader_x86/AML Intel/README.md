10-12-2024 - I wasn't going to upload it anymore, but I want to have it here in 2024 to start delving into the topic. Like most, this is not a topic for 1 A4 page as the Readme.md file, but...<br /><br />

1. Download iASL Compiler and Windows ACPI Tools (.zip, 1.3 MB) toolchain -  Get https://www.intel.com/content/www/us/en/developer/topic-technology/open/acpica/download.html 
 --- https://github.com/user-attachments/files/17171016/iasl-win-20240927.zip<br />
2. Here is quick introduce to ASL [ACPI Source Language (ASL) Tutorial] https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiozI_li56KAxVrHRAIHW64GvgQFnoECBoQAQ&url=https%3A%2F%2Fcdrdv2-public.intel.com%2F772722%2Fasl-tutorial-v20190625.pdf&usg=AOvVaw3oEVdOF0XZO01rL0uN_9go&opi=89978449<br />
3. Main repo https://github.com/acpica/acpica/ <br />
4. OSDev wiki AML https://wiki.osdev.org/AML
<hr>
1. First of all, there are several tools built into the Windows environment. https://learn.microsoft.com/en-us/windows-hardware/drivers/devtest/devcon - This will display all enumerated devices in system, not just ACPI-based ones. >> " devcon find * " <br />
2. Using registry to find devices - HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Enum\ to look at the tree structure and find specific devices like BTH, HID, HDAUDIO, etc<br />
3. For AML this is part of ACPI Devices: Managed by the operating system using ACPI methods to manage power and configuration (e.g., thermal zones, battery status, power buttons). But apart from that there is PnP Devices: These are managed by the PnP manager, which can dynamically add or remove devices from the system and assign system resources such as IRQs, I/O ports, and memory addresses. The Plug and Play manager will enumerate devices, assign drivers, and configure devices.<br />
<hr>
According to information from the web it can be loaded into qemu or built .aml in Tianocore EDK2. | "Override with Driver-Based Solutions: On Windows, overriding ACPI tables requires writing a custom kernel driver to intercept and load your modified ACPI table. This involves advanced kernel development and is not officially supported for end-users." But this is only information. TODO.


```
qemu-system-x86_64 -m 2G -enable-kvm \
    -acpitable file=dsdt.aml \
    -hda your_disk_image.img
```
    
<hr>
<h2>OK, what is on these pictures?</h2>
This is guide for Windows. There is similar version for linux, but using linux commands <br />
<br /><br />


Workflow Example: Modify and Test ACPI Tables

Step 1: Extract ACPI Tables. Use acpidump to dump the ACPI tables. This creates .dat files like dsdt.dat for the DSDT table. <b>This command get ACPI table from my host machine probably, this is ASUS laptop for ma case</b>

```
acpidump -b 
```

Step 2: Disassemble the Table. Disassemble the table to human-readable ASL format. This generates a file, dsdt.dsl, containing the ASL source code.

```
iasl -d dsdt.dat
```

Step 3: Modify the ASL Source. 
<br />
Example placement:

```
Device (TEST)
{
    Name (_HID, "ACPI0001")  // Example hardware ID for the device
    Method (_TST, 0, NotSerialized)
    {
        Return (42)  // Dummy return value
    }
}
```

Or, if global: (<b>I used this like you see on last picture</b>)

```
Method (_TST, 0, NotSerialized)
{
    Return (42)  // Dummy return value
}
```

Compile (Recompile the Modified ASL command -tc argument). <b>In this step, the output file is to be created as .AML. All errors need to be corrected. Use "iasl -ve -tc dstd.dsl to examine and print only errors message, without warnings and remarks. In my case. I was 2 errors and the correction was to change the size of the letters pnp0c14 to PNP0C14 etc. -- Non hex letters must be upper case Error ---  </b>

```
iasl -tc dsdt.dsl
```

Test with acpiexec

```
AcpiExec -f dsdt.aml // I used without -f argument
```

Run

```
execute \_TST // or evaluate \_TST
```

Expected output

```
Method returned: Integer 42
```

As I wrote. This is not a topic for an A4 size Readme. This is a quick intro.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/AML%20Intel/123%20-%2010-12-2024%20-%20evaluate%20i%20execute%20to%20chyba%20to%20samo.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/AML%20Intel/127%20-%2010-12-2024%20-%20cd.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/AML%20Intel/128%20-%2010-12-2024%20-%20cd.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/AML%20Intel/119%20-%2010-12-2024%20-%20ok%20.png?raw=true)

<hr>
Sending to search engine GOOGLE > "EDK2 .aml" or "EDK2 intel package asl" - first results refer to tianocore. For example https://github.com/tianocore/edk2-platforms/blob/master/Platform/Intel/Readme.md | https://github.com/tianocore/edk2-platforms/blob/master/Platform/Intel/KabylakeOpenBoardPkg/Acpi/BoardAcpiDxe/Dsdt/CPU.asl
