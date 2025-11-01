Same as in the previous demos - source code and ```editor.efi``` file
<br /><br />
Ok, you can see pictures of this in "demo1 - shell demo fixes". This is a simple program that will read logs from /EFI/Boot/myLogs which are on a USB flash drive.
<br /><br />
This isn't a text editor. It just reads the logs as shown in the images, and you can scroll the buffer up and down with the PAGE UP and PAGE DOWN keys. This allows you to avoid using the serial port, etc., and just collect the logs and read them immediately.

<h3>How to run</h3>
1. Just put it in /EFI/Boot/myApps/<br />
2. So the command "listapps" sees this in folder<br />
3. Then loading via "loadimg editor.efi"<br />
4. Program lists the logs and waits for the index to be entered<br />
5. Press enter based on the information in the terminal and enter to log reader
<br /><br />

![dump](https://github.com/KarolDuracz/scratchpad/raw/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes/images/1762021891901.jpg?raw=true)
