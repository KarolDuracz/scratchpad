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

<b>For BIOS (MBR)</b>

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
Locate the Windows Installation Image:

Typically, this is in the install.wim or install.esd file on the Windows installation media (e.g., X:\sources\install.wim where X: is the drive letter for your installation media). 
Use DISM to Apply the Image:

Use the DISM tool to apply the Windows image from the installation media to the C: drive (the Windows partition you created).

```
dism /apply-image /imagefile:X:\sources\install.wim /index:1 /applydir:C:\
```

Replace X: with the drive letter of the installation media, and adjust the /index if you want to install a different edition (you can list available editions in install.wim using dism /get-wiminfo).
<br /><br />
But im my case I used. Because when you find CDROM drive, and for me that is D:\ this *.wim file there is no name install.wim, only boot.wim when you use DIR to list this D:\ drive. And you find in D:\sources\ 

```
dism /apply-image /imagefile:D:\sources\boot.wim /index:1 /applydir:C:\
```
<h2>Configure the Bootloader</h2>
Create Boot Files:

Run bcdboot to create the necessary boot files on the System partition

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
