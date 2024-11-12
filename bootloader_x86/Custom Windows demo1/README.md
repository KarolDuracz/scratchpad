<h2>What Windows PE is?</h2> 
Here is a quick introduce of what is Windows PE <br />
https://learn.microsoft.com/en-us/windows-hardware/manufacture/desktop/winpe-intro?view=windows-11
<hr>

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/Custom%20Windows%20demo1/135%20-%209-11-2024%20-%20custom%20windows%20with%20a%20minimal%20file%20system%20and%20tools.png?raw=true)

Sorry, but I can't write anything more here for now. I managed to compile a custom Windows kernel based on ver 10.0.26100.1 in yet another attempt after few weeks. After packing this to .ISO image is 386 MB. But there is only a desktop, no explorer.exe in this minimal configuration. There is no calc.exe etc. etc. there is not even "findstr" to filter the search in CMD. Very minimalist system to start build something from scratch :)
<br /><br />
... 
<br />
After installing ADK 10.1.26100.1 and Windows PE add-on for the Windows ADK 10.1.26100.1 (May 2024) there are only a few commands 5-6 steps and in a few minutes you can build such an ISO image which works as you can see. But I haven't tested it on real hardware yet. I just managed to run it today. Finally. I still need to review step by step what I did and what I got. I tried to manually copy the files required by UEFI, paths and folders. But this requires configuring bootmgr I think. Nothing came of it. Only installing ADK 10.1.26100.1 worked for me. FINALLY.
<br/><br />
Last time I tried to open such a file system and nothing else was there. It was possible to create an image but it did not boot in VirtualBox. And I did not know how to verify it... but I extracted these files from live CD etc. Without any configuration.
```
minimal_win_iso/
├── Boot/
│   ├── BCD               # Boot Configuration Data
│   ├── boot.sdi          # (Optional, used in recovery environments)
│   ├── bootmgr           # Windows Boot Manager (Optional, if using custom bootloader, skip this)
├── EFI/
│   ├── Boot/
│       ├── bootx64.efi    # EFI bootloader (can be winload.efi)
└── Windows/
    ├── System32/
        ├── ntoskrnl.exe   # Windows kernel
        ├── hal.dll        # Hardware Abstraction Layer
        ├── winload.exe    # Windows loader (optional, used if bootmgr)

```

<br/><br />
to be continued...

https://learn.microsoft.com/pl-pl/windows-hardware/get-started/adk-install <br />
https://learn.microsoft.com/en-us/windows-hardware/manufacture/desktop/winpe-create-usb-bootable-drive?view=windows-11 <br />

<br />
<hr>
<br />
<h2>Step 1:</h2> Install Windows ADK with Windows PE Add-on <br />
Download and Install Windows ADK: <br />
Ensure that you have installed the Windows ADK along with the Windows PE add-on. You’ll need this add-on to create the minimal Windows PE environment. <br />
 <br /> 
Open Deployment and Imaging Tools Environment:<br />
Open the Deployment and Imaging Tools Environment (a command prompt environment) with Administrator privileges.<br /><br />
https://learn.microsoft.com/en-us/windows-hardware/get-started/adk-install
<br /><br />
I installed last version ADK 10.1.26100.1 (May 2024)
<br /><br />
<h2>Step 2 : Create a Windows PE Working Directory</h2> This directory will contain the initial Windows PE image that you’ll customize. But dont create any folders inside. Follow this command. This command copy and create, IF NOT EXIST (if exists this directory inside that destination folder then return error about that), command will create a folder called WinPE_amd64 in C:\ with a minimal Windows PE boot image
<br /><br />
Verify the Files Created: The directory should contain a media folder with boot.wim, EFI, and Boot directories. This structure will form the foundation of the minimalist image
<br /><br />
Open as Administrator

``` 
C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Windows Kits\Windows ADK\Deployment and Imaging Tools Environment 
```

And run this command

```
copype amd64 C:\usr_bin\customWindows_demo2\mount
```
<br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/Custom%20Windows%20demo1/pics/step%201%20-%20copy%20file.png?raw=true)

<h2>Step 3 : Mount the Windows PE Image</h2> This mounts the Windows PE image to C:\usr_bin\customWindows_demo2, allowing you to make modifications.
<br />
Inside C:\usr_bin\customWindows_demo2 crete another folder, for example offline in my case

```
CMD > cd C:\usr_bin\customWindows_demo2
CMD > mkdir offline
```

Open and run DISM.exe as Administator for AMD64

```
C:\Program Files (x86)\Windows Kits\10\Assessment and Deployment Kit\Deployment Tools\amd64\DISM\dism.exe
```

Type and run this command

```
C:\Program Files (x86)\Windows Kits\10\Assessment and Deployment Kit\Deployment Tools\amd64\DISM>DISM /Mount-Image /ImageFile:C:\usr_bin\customWindows_demo2\mount\media\sources\boot.wim /Index:1 /MountDir:C:\usr_bin\customWindows_demo2\offline
```

<br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/Custom%20Windows%20demo1/pics/step%202%20-%20create%20offline%20folder%20and%20follow%20dism%20command.png?raw=true)

<h2>Step 4 : Customize the Minimal Windows PE Image</h2> At this stage, you have a basic Windows PE image mounted, and you can add or remove components based on your requirements. Here are some possible customizations:
<br /><br />
Example commands : Remove Unnecessary Packages
<br /><br />
List available packages to see what’s installed by default

```
DISM /Image:C:\WinPE_Mount /Get-Packages
```

To remove unnecessary packages and reduce the image size, use the following command, replacing PackageName with the name of the package you want to remove:

```
DISM /Image:C:\WinPE_Mount /Remove-Package /PackageName:<PackageName>
```

Example commands : Add Essential Files Only
<br /><br />

By default, Windows PE includes basic components. To make it more minimal, avoid adding additional drivers or applications unless necessary.

If you need specific drivers, add them as follows:

```
DISM /Image:C:\WinPE_Mount /Add-Driver /Driver:C:\Path\To\Driver /Recurse
```
<br />
Example commands : Add Scripts or Automation Tools
<br /><br />
If you need basic automation or setup scripts, copy them into the mounted image

```
copy C:\Path\To\YourScript.bat C:\WinPE_Mount\Windows\System32\
```

<h1>I SKIPPED THIS PART #4 !!!</h1> 

<br />
<h2>Step 5 : Set Up Boot Configuration for Minimal Boot</h2> Windows PE is already set up with minimal boot files, but you can customize its boot configuration:
<br /><br />
Configure Boot Settings:
Windows PE boots to X:\ by default, which is a RAM disk. Modify winpeshl.ini to control startup programs. 
Save this configuration to run the command prompt by default upon boot.

```
[LaunchApp]
AppPath = %SYSTEMROOT%\System32\cmd.exe
```

<br />
<h1>I SKIPPED THIS PART  #5 !!!</h1> 
<br />
<h2>Step 6 : Commit Changes and Unmount the Image</h2> After you’ve customized the image, commit the changes to the boot.wim file

```
DISM /Unmount-Image /MountDir:C:\usr_bin\customWindows_demo2\offline /Commit
```
This is tricky. The boot.wim file is saved in the media\sources folder of your Windows PE working directory after you unmount and commit the image

```
C:\WinPE_amd64\media\sources\boot.wim
```

But for this command : DISM /Unmount-Image /MountDir:C:\usr_bin\customWindows_demo2\offline /Commit <-- we need this source offline, not first mounted folder. But second from step 3.
<br/><br />
After this step folder C:\usr_bin\customWindows_demo2\offline is empty !
<br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/Custom%20Windows%20demo1/pics/step%206%20-%20cd.png?raw=true)

<h2>Step 7 : Create a Bootable ISO (Minimal Windows PE Image)</h2> Once your minimal Windows PE environment is configured, you can create a bootable ISO:
<br />
Generate the ISO using the MakeWinPEMedia command. First Open next CMD prompt as Administrator (but this is not necessary)

```
MakeWinPEMedia /ISO C:\usr_bin\customWindows_demo2\mount C:\usr_bin\customWindows_demo2\minimalWinPE_demo2.iso
```


```
C:\Program Files (x86)\Windows Kits\10\Assessment and Deployment Kit\Windows Preinstallation Environment>MakeWinPEMedia /ISO C:\usr_bin\customWindows_demo2\mount C:\usr_bin\customWindows_demo2\minimalWinPE_demo2.iso
Creating C:\usr_bin\customWindows_demo2\minimalWinPE_demo2.iso...

100% complete

Success


C:\Program Files (x86)\Windows Kits\10\Assessment and Deployment Kit\Windows Preinstallati
on Environment>
```
<br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/Custom%20Windows%20demo1/pics/step%207%20-%20final%20step.png?raw=true)


<br />

<h2>Summary of Commands</h2>

```
# 1. Create the Windows PE working directory
copype amd64 C:\WinPE_amd64

# 2. Mount the image for customization
mkdir C:\WinPE_Mount
DISM /Mount-Image /ImageFile:C:\WinPE_amd64\media\sources\boot.wim /Index:1 /MountDir:C:\WinPE_Mount

# 3. Customize the image (optional)
DISM /Image:C:\WinPE_Mount /Get-Packages
DISM /Image:C:\WinPE_Mount /Remove-Package /PackageName:<PackageName>

# 4. Commit and unmount the image
DISM /Unmount-Image /MountDir:C:\WinPE_Mount /Commit

# 5. Create a bootable ISO
MakeWinPEMedia /ISO C:\WinPE_amd64 C:\WinPE_amd64\MinimalWinPE.iso
```

<br />
<hr>
<br />
Many many tools not exists in that minimalist image.

```
net use Z: \\192.168.1.100\customwin_demo1 /user:admin password123 // for now I can't setup this
netstat -an | find "445" // SMB status not exists
netsh interface ipv4 set dns name="Local Area Connection" static <DNS_IP_Address> // netsh works but...
netsh interface ipv4 set global netbios=enabled // netbios command is not exist
net view // in here username is empty !!!
net user Guest /active:yes // command to create guest user for anonymous connection
net user Guest // ...
Enter the user name for '192.168.1.102': Guest // ...
Enter the password for '192.168.1.102': [Press Enter if blank] // ...
ipconfig // it works ok
ping 192.168.1.102 // ping to host machine works
wpeinit // Run wpeinit to initialize networking. This is start after system console is booted.
net start dhcp // it run after wpeinit - it works in backround
// I have an some errors with
diskpart // open diskpart
list disk
list volume
wmic diskdrive get caption,deviceid,model,size // wmic doesn't exists in this image
net use Z: \\VBOXSVR\Shared_Folder // this command not working can't handle conection 
dir \\VBOXSVR\Shared_Folder // not working can't handle conection 
bitsadmin /transfer myDownloadJob /download /priority normal https://example.com/file.txt C:\file.txt // this command doesn't exists in this minimalist image
```

Many things is to learn but next thing is to customize image for example by uploading CURL etc tools and then creating an ISO image with these tools.
<br />
<hr>
https://learn.microsoft.com/pdf?url=https%3A%2F%2Flearn.microsoft.com%2Fen-us%2Fwindows-hardware%2Fmanufacture%2Ftoc.json%3Fview%3Dwindows-11
