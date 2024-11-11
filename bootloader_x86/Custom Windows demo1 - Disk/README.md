For this step we need
<br /><br />
1. Installed Windows PE as a bootable CD disk - https://github.com/KarolDuracz/scratchpad/tree/main/bootloader_x86/Custom%20Windows%20demo1
<br /><br />
2. Working FTP -  https://github.com/KarolDuracz/scratchpad/tree/main/bootloader_x86/Custom%20Windows%20demo1%20-%20FTP
<br /><br />
3.  (we are here) Initialize and Partition the Disk. Then, we will create an application for this WinPE with MSVC 2019 for tests.
<hr>
For this step we find guid in these links (look at left menu and topics in links)<br />
https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-8.1-and-8/hh824839(v=win.10)<br />
https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-8.1-and-8/hh825089(v=win.10)<br />
<br />

1. Open DiskPart to manage disk partitions

```
diskpart
```

2. List all disks to identify the target disk (usually Disk 0 for a single-disk setup)

```
list disk
```

3. Select the target disk (replace 0 with the appropriate disk number if necessary)

```
select disk 0
```

4. Clean the disk to remove any existing partitions

```
clean
```

5. Create a new partition (usually an EFI partition for UEFI or an MBR for BIOS) - <b> for UEFI </b>

```
convert gpt
create partition primary size=100
format fs=fat32 quick label="System"
assign letter=S

create partition primary
format fs=ntfs quick label="Windows"
assign letter=C
```

<b>For BIOS (MBR)</b> // I don't use these commands. But it is important to know that. I built for UEFI. 

```
convert mbr
create partition primary
format fs=ntfs quick label="Windows"
active
assign letter=C
```

6. Exit DiskPart after partitioning

```
Exit
```
<hr>
<h2>Apply the Windows Image to the New Partition</h2>
<br />

Use this command to find DVD_ROM / CD_ROM
<br />

```
CMD > diskpart
diskpart > list volume
```

<br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/Custom%20Windows%20demo1%20-%20Disk/step%20x%20-%20find%20cd%20rom%20using%20diskpart.png?raw=true)

Locate the Windows Installation Image:

Typically, this is in the install.wim or install.esd file on the Windows installation media (e.g., X:\sources\install.wim where X: is the drive letter for your installation media). 
Use DISM to Apply the Image:

Use the DISM tool to apply the Windows image from the installation media to the C: drive (the Windows partition you created).

```
dism /apply-image /imagefile:X:\sources\install.wim /index:1 /applydir:C:\ // this is example script
```

Replace X: with the drive letter of the installation media, and adjust the /index if you want to install a different edition (you can list available editions in install.wim using dism /get-wiminfo).
<br /><br />
But im my case I used this. Because when you find CDROM drive using diskpart, and for me that was D:\ , this *.wim file is not named "install.wim", but "boot.wim". And when you use DIR command to list CD-ROM D:\ drive. You find it in D:\sources\ 

<h3>1</h3>

```
dism /apply-image /imagefile:D:\sources\boot.wim /index:1 /applydir:C:\
```
<h2>Configure the Bootloader</h2> 
Create Boot Files:

Run bcdboot to create the necessary boot files on the System partition

<h3>2</h3>

```
bcdboot C:\Windows /s S: /f UEFI
```
Replace /f UEFI with /f BIOS if using BIOS mode, and adjust the drive letters if necessary.
Verify the boot setup:

For UEFI, ensure that the EFI partition is labeled as such and contains the necessary boot files.
For BIOS, make sure the Windows partition is marked as active in DiskPart (done previously).

<h2>Restart and Boot into the New Installation</h2>
Remove the Windows PE Media:

Eject the USB drive or unmount the ISO from the virtual machine to avoid booting into Windows PE again.
Reboot the Machine:

<h3>3</h3>

```
Wpeutil Reboot
```

The system should now boot into the Windows installation on the hard drive, proceeding through any first-time setup steps. Without this step you need press key to boot CD and then you jump to Windows PE. But after this installation you don't press any keys. Wait. This is scenario similar to this, when the BIOS is set CD/DVD as the first boot device but you don't click to enter the installer, you just wait for the system to load. The same thing happens after creating a disk this way. You wait for the system to load.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/Custom%20Windows%20demo1%20-%20Disk/step%201%20-%20create%20partition.png?raw=true)

<hr>

<h2>After booting live CD this path to install.wim doesn't exists X:\sources\install.wim</h2>
If you're booting from a Windows PE environment and don’t have access to the install.wim file, this could mean that either:

The installation files (including install.wim) are not present on the boot media, or
You are using a custom Windows PE image that does not include the installation files.
Steps to Access install.wim or install.esd
Option 1: Check the Installation Media
If you’re using a Windows installation ISO, you should be able to find install.wim or install.esd on the installation media, typically located in:

```
[ISO or USB Drive Letter]:\sources\install.wim
```

Option 2: Mount the Windows Installation ISO in the Virtual Machine or Physical Machine
If install.wim is not present on the Windows PE media itself, you can do the following:

Mount the Windows Installation ISO on the virtual machine or physical machine:

If using VirtualBox, open the virtual machine settings, navigate to Storage, and add the Windows installation ISO to the CD/DVD drive.
If you're on a physical machine, insert the Windows installation USB or DVD with the full installation files.
Find the Drive Letter of the Mounted ISO:

After mounting, return to your Windows PE command prompt.
Use diskpart to identify the drive letter of the newly mounted ISO

```
diskpart
list volume // <--- this line find CDROM device . For me that was D:\
exit
```

Look for a volume labeled "DVD-ROM" or similar, which represents the Windows installation media. Locate the install.wim or install.esd File:

Check if the file is available in [Drive Letter]:\sources\install.wim or [Drive Letter]:\sources\install.esd.
Apply the Image with DISM:

Once you've identified the drive letter, use the DISM command to apply the image to your hard drive as shown previously:

```
dism /apply-image /imagefile:[Drive Letter]:\sources\install.wim /index:1 /applydir:C:\
```

Replace [Drive Letter] with the actual drive letter of the Windows installation ISO.
<hr>
<b>That's it. Now we have "installed" Windows PE on disk. Just Wait and watch "Press any key to boot from CD or DVD....". </b>
<hr>

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/Custom%20Windows%20demo1%20-%20Disk/step%202%20-%20try%20to%20run%20some%20apps.png?raw=true)

When I moved directly for tests some application, first try is to move calc.exe and explorer.exe from HOST machine (win 8.1). And in this environment there is no error information. Silence. The process simply was not created. Then I did some tests using my demo versions from the \win32\ repository folder. And I got some information "use Sxstrace (Side-by-Side) ... something's". And in this Windows PE this tool exists. And on this image you see log from this tool.
<br /><br />
Run this command to start the trace:
```
start cmd.exe // open new window with cmd
sxstrace.exe trace -logfile:C:\sxstrace_log.etl

```
To stop CTRL + C or press ENTER key. Alternatively, you can stop it by running

```
sxstrace.exe stoptrace
```

Convert the Trace Log to a Readable Format

```
sxstrace.exe parse -logfile:C:\sxstrace_log.etl -outfile:C:\sxstrace_log.txt
```

When this tool worked in background on the other instance of CMD, I used FTP configuration from previous folder (https://github.com/KarolDuracz/scratchpad/tree/main/bootloader_x86/Custom%20Windows%20demo1%20-%20FTP) to upload some applications for tests. Then I stopped sxstrace, conver log to .txt  and opened :

```
notepad.exe C:\sxstrate_log.txt
```
<hr>

From this sxstrace I found out that it is trying to run a win32 application and this minimalist Windows PE can't handle this for some reasons. It's not even about the fact that the application was 32 bit. Because 64 bit didn't work either. Nothing worked, except Process Explorer which was visible in the previous pictures.
<br /><br />
the reason why it doesn't work is that it requires dependencies that Windows PE doesn't have, i.e. the DLL files you see here, like VCRUNTIME14.dll, ucrtbased.dll etc.
<br /><br />
Command to check this:

```
dumpbin /dependents yourapp.exe
```

 ![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/Custom%20Windows%20demo1%20-%20Disk/step%203%20-%20this%20is%20reason%20why%20this%20not%20working.png?raw=true)

<hr>
<h2>The solution</h2>
<br/>
1. Is to upload these .dll files into Windows PE file system. Reboot system and try run. <b></b>But I haven't tried it in this approach.</b> .  
<br/><br/>
2. For my case, I want to run custom application. 
<br/><br/>
For me, the setting helped<br />
Under Configuration Properties > C/C++ > Code Generation, find the Runtime Library option. <br />
<b></b>Select Multi-threaded (/MT) instead of Multi-threaded DLL (/MD).</b> <br />
/MT: Statically links the C runtime into your executable. This eliminates the dependency on VCRUNTIME140.dll. <br />
/MD: Dynamically links to the C runtime (the default), which is what brings in VCRUNTIME140.dll. <br />
<br /><br />
Disable Exception Handling and RTTI (if Applicable) <br />
Under Configuration Properties > C/C++ > Code Generation <br />
Set Enable C++ Exceptions to No (unless you need exceptions).
<br/><br/>

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/Custom%20Windows%20demo1%20-%20Disk/step%204%20-%20build%20apps.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/Custom%20Windows%20demo1%20-%20Disk/step%204%20-%20another%20pic.png?raw=true)


<h2>Guide from chat gpt how to configure MSVC 2019 for this case</h2>
<br/><br/>
Building a project for Windows PE (WinPE) AMD64 using MSVC (Microsoft Visual Studio 2019) involves setting up a build configuration that produces binaries compatible with the minimalistic Windows PE environment. This setup requires targeting the 64-bit (AMD64) architecture and minimizing dependencies that Windows PE might not support.

Here are the steps to set up and build a project in MSVC 2019 for Windows PE AMD64:

1. Create or Open Your Project in Visual Studio
Open Visual Studio 2019 and either:

Open an existing project.
Create a new project targeting Windows Desktop Application or Console Application.
Ensure that you select C++ (if applicable) or another suitable language for the project.

2. Configure Project Properties for Windows PE Compatibility
To ensure compatibility with the WinPE environment, you’ll need to adjust some build settings in the project’s configuration:

Open Project Properties:

Right-click on the project in Solution Explorer.
Select Properties.
Set Target Platform:

Under Configuration Properties > General, set:
Platform Toolset to a recent version (e.g., Visual Studio 2019 (v142)).
Configuration Type to Application (.exe) or Dynamic Library (.dll), depending on your needs.
Target Platform to x64 (for AMD64 compatibility).
Disable Unsupported Windows Features:

Go to Configuration Properties > Linker > System and set Subsystem to Console (if it’s a console application) or Windows (for GUI applications).
In Configuration Properties > Linker > Advanced, set Entry Point to main (or your specific entry point) if it’s a console application.
Optimize for Minimal Dependencies:

Avoid using libraries and dependencies that are unavailable in Windows PE.
Under Configuration Properties > C/C++ > Code Generation:
Set Runtime Library to Multi-threaded (/MT) or Multi-threaded DLL (/MD) if you’re including runtime DLLs. Using /MT statically links the runtime, which is helpful if you want fewer dependencies.
Avoid using features that require .NET Framework, COM, or MFC/ATL, as these are generally unsupported in Windows PE.
Add Conditional Compilation for Windows PE:

You may want to add conditional compilation definitions specific to Windows PE.
Go to Configuration Properties > C/C++ > Preprocessor and add WINPE as a Preprocessor Definition to allow for any platform-specific code you might need.
Minimize Dependencies:

Limit dependencies to essential components. Many libraries (such as .NET or third-party DLLs) may not be available in Windows PE.
3. Build the Project
Build the Project:

Set the Solution Configuration to Release and Solution Platform to x64.
Go to Build > Build Solution (or press Ctrl+Shift+B).
This generates an AMD64-compatible binary in the project’s Release folder, suitable for running on Windows PE.
Verify Dependencies:

After building, use the Dependency Walker (depends.exe) or the dumpbin utility to verify that your binary doesn’t rely on unsupported DLLs (like user32.dll, gdi32.dll, etc.).

4. Deploy and Test on Windows PE
Copy the Binary to Windows PE:

Copy your built .exe or .dll files to the Windows PE environment, either by including them in a bootable WinPE ISO or mounting the WinPE image with DISM and adding the files.
Boot into Windows PE and Run the Application:

Boot into Windows PE and open a command prompt.
Navigate to the folder where your application is located and run it.
Additional Considerations
Dependency Issues: If your application fails to run, check for missing dependencies. Windows PE is very limited in its available components.
Using Visual C++ Redistributables: If your app requires Visual C++ runtime libraries, copy the appropriate runtime DLLs (like msvcrt.dll) directly into the same folder as your executable in Windows PE, or link the runtime statically by using /MT in the Runtime Library setting.
This setup ensures you’re creating a minimal, AMD64-compatible binary for Windows PE using MSVC 2019.
<br />
If you want to remove dependencies on the Visual C++ runtime library (VCRUNTIME140.dll) and rely only on basic libraries such as kernel32.dll (for minimal, low-level functions) while still using stdio.h functions in your project, you need to adjust your Visual Studio project to avoid linking against the Visual C++ runtime.

Here's how to achieve this:

1. Set Runtime Library to /MT (Static Linking)
By statically linking the runtime, you can avoid requiring VCRUNTIME140.dll and other dynamic runtime libraries.

Open your project in Visual Studio.

Go to Project Properties (right-click on the project in Solution Explorer and select Properties).

Under Configuration Properties > C/C++ > Code Generation, find the Runtime Library option.

Select Multi-threaded (/MT) instead of Multi-threaded DLL (/MD).

/MT: Statically links the C runtime into your executable. This eliminates the dependency on VCRUNTIME140.dll.
/MD: Dynamically links to the C runtime (the default), which is what brings in VCRUNTIME140.dll.
Click OK to save your changes.

2. Avoid C++ Standard Library Features That Pull in Extra Dependencies
The stdio.h library functions can typically be linked with kernel32.dll and msvcrt.lib (if statically linked), so as long as your application sticks to basic C functions like printf, fopen, etc., you can avoid heavier runtime dependencies.

However, if your code uses C++ standard library features, you may inadvertently pull in dependencies on additional runtime libraries. Avoid C++ standard libraries (such as iostream, <string>, <vector>, etc.) if possible, and stick to C-style code (stdio.h, stdlib.h, etc.) when aiming for minimal dependencies.

3. Disable Exception Handling and RTTI (if Applicable)
To further reduce dependencies:

Under Configuration Properties > C/C++ > Code Generation:
Set Enable C++ Exceptions to No (unless you need exceptions).
Set Runtime Type Information (RTTI) to No.
These settings reduce dependencies on runtime support code that Visual C++ otherwise includes.

4. Rebuild and Test
After making these adjustments, rebuild your project. Your output executable should no longer have dependencies on VCRUNTIME140.dll or other Visual C++ runtime DLLs.

5. Verify Dependencies
Use dumpbin or Dependency Walker to verify that your executable only depends on core Windows libraries (kernel32.dll, msvcrt.dll, etc.):

```
dumpbin /dependents yourapp.exe
```

This command will show you the exact DLL dependencies of your executable. Ideally, you should see only minimal dependencies, such as kernel32.dll and msvcrt.dll (if you used standard C libraries).

Summary
By using /MT for static linking, removing C++ standard library features, and disabling exception handling and RTTI, you can create a minimal executable that avoids VCRUNTIME140.dll and uses only basic system libraries like kernel32.dll for stdio.h functionality. This approach works well in environments like Windows PE, where runtime dependencies should be minimal.
<hr>
<h2>Summary for this part</h2>
Ok, I still haven't touched on the topic of building a custom Windows from, i.e. adding drivers, uploading custom programs like CURL, etc. This is more advanced topic for me right now. This is a topic for the next part no. 4 but it will take some time. At this point my goals have been achieved in a sense when it comes to building a custom windows (minimalist windows kernel).
<br/><br />
now I can just type

```
Wpeutil Shutdown
```

And when I press ENTER to shutdown virtual machine guest system, all files next time when I turn on, they will still be on the disk.
<br /><br />

UPDATE - 11-11-2024 - 
