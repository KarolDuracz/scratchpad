![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/Custom%20Windows%20demo1/135%20-%209-11-2024%20-%20custom%20windows%20with%20a%20minimal%20file%20system%20and%20tools.png?raw=true)

Sorry, but I can't write anything more here for now. I managed to compile a custom Windows kernel based on ver 10.0.26100.1 in yet another attempt after few weeks. After packing this to .ISO image is 386 MB. But there is only a desktop, no explorer.exe in this minimal configuration. There is no calc.exe etc. etc. there is not even "findstr" to filter the search in CMD. Very minimalist system to start build something from scratch :)
<br /><br />
... 
<br />
After installing ADK 10.1.26100.1 and Windows PE add-on for the Windows ADK 10.1.26100.1 (May 2024) there are only a few commands 5-6 steps and in a few minutes you can build such an ISO image which works as you can see. But I haven't tested it on real hardware yet. I just managed to run it today. Finally. I still need to review step by step what I did and what I got. I tried to manually copy the files required by UEFI, paths and folders. But this requires configuring bootmgr I think. Nothing came of it. Only installing ADK 10.1.26100.1 worked for me. FINALLY.
<br/><br />
to be continued...

https://learn.microsoft.com/pl-pl/windows-hardware/get-started/adk-install <br />
https://learn.microsoft.com/en-us/windows-hardware/manufacture/desktop/winpe-create-usb-bootable-drive?view=windows-11 <br />
