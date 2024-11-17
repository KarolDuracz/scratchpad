4-11-2024 - This is only beginning. But this might be helpful for running few CMD in one window. But here you see 3 mspaint.exe running inside MIDI under parent window. Under construction.<br /><br />

17-11-2024 - In sysinternals (https://learn.microsoft.com/en-us/sysinternals/downloads/desktops) tools these is Desktops64.exe . And this app correspond to Desktop.exe in some way. But this is just the beginning of the demo.<br /><br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Multi%20Windows%20Manager/71%20-%204-11-2024%20-%20win32%20%20multiple%20window%20manger.png?raw=true)

<h2>Guide, how it works in simple way.</h2>

Create "NEW" window from top menu, and when MIDI window show inside parent window click "OK". 
<br /><br />
1. In file https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Multi%20Windows%20Manager/Clipboard_popupWindow/Clipboard_popupWindow/Clipboard_popupWindow.cpp
   in lines 166 to 171 is that

```
    if (CreateProcess(
        //L"C:\\Windows\\System32\\calc.exe", // Path to calc.exe
        //L"C:\\Program Files (x86)\\Windows Media Player\\wmplayer.exe",
       // L"C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
       // L"C:\\Windows\\System32\\notepad.exe",
       // L"C:\\Windows\\System32\\cmd.exe",
        L"C:\\Windows\\System32\\mspaint.exe",
```
This is few example of path to process but this is tricky, because we need also window name. Examples from 185 line
```
        // Get the handle of the Calculator's main window
       // HWND hCalcWnd = FindWindow(NULL, L"Windows PowerShell");
      //  HWND hCalcWnd = FindWindow(NULL, L"Untitled - Notepad");
       // HWND hCalcWnd = FindWindow(NULL, L"C:\\Windows\\System32\\cmd.exe");
        HWND hCalcWnd = FindWindow(NULL, L"Untitled - Paint");
        if (hCalcWnd) {
```

So, for CMD.exe as path we need "L"C:\\Windows\\System32\\cmd.exe"," but as windows name "HWND hCalcWnd = FindWindow(NULL, L"C:\\Windows\\System32\\cmd.exe");"
<br /><br />
Under construction.
<br /><br />
https://nexus-6.uk/joomla/index.php/multiple-document-interface-menu-item <br />
https://learn.microsoft.com/en-us/windows/win32/winmsg/about-the-multiple-document-interface

<hr>

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Multi%20Windows%20Manager/8%20-%201-11-2024%20-%20cdc.png?raw=true)

First time, I try to creted this without MIDI, but it didn't work out right. Then I switched to the midi version. Because I have guide from this links. But I'll definitely put it here sometime. I just need to spend more time working out the basic functions. 

<h2> This is worth to development, it's easy to manage like that than show from task bar all windows and select as popups. Better sometimes is running 4 CMD or something under 1 window.</h2>
