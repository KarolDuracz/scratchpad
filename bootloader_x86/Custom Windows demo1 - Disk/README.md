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

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/Custom%20Windows%20demo1%20-%20Disk/step%201%20-%20create%20partition.png?raw=true)

