One more thing. (btw. something new will appear from these things every now and then, but this example is interesting and worth improving). Yesterday I wrote that uploading it further does not make sense, but one more thing to complete the set with demo2 and demo3.
<br /><br />
![dump](https://raw.githubusercontent.com/KarolDuracz/scratchpad/refs/heads/main/Win32/Simple_debugger/demo4/80%20-%205-11-2024%20-%20debug%20techniqe%203%20demo.png)

<h2>How to run</h2>
1. Inside folder there is "compile.txt" file with command to compile via cl.exe<br />
2. Run SimpleGUI.exe<br />
3. Run debugger.exe <br />
4. Pass as parameter PID of process to SimpleGUI.<br />
5. You see result like on image<br />
<br /><br />
The main goal is to trace scenario like this (for example using WinDbg):<br />
1. ? rcx + 0x8 // identify the addres <br />
2. ba r4 0x00007FF683590008 ".if (poi(0x00007FF683590008) == 201) { .echo Condition met: *(RCX + 0x8) == 201; }" // setup break point on address with contion RCX + 8 == 201 this is for Left mouse click on window <br />
or <br />
bp user32!DispatchMessageW ".if (poi(@rcx+0x8) == 201) { .echo Condition met: RCX+0x8 == 201; }"<br />
3. bp user32!DispatchMessageW // setup bp on this function <br />
4. dt user32!tagMSG @rcx   // 64-bit systems<br />
or dt user32!tagMSG poi(esp+4) // 32 bit system<br />
5. When "Symbol user32!tagMSG not found." you'll need
6. ? rcx // address to  this structure

```
user32!tagMSG
   +0x000 hwnd            : 0x12345678 HWND
   +0x008 message         : 0x201 (WM_LBUTTONDOWN)
   +0x010 wParam          : 0x1
   +0x018 lParam          : 0x20001
   +0x020 time            : 123456789
   +0x024 pt              : _POINT ( X = 0x100, Y = 0x200 )
```
<br /><br />
This must solve scenario when user KNOW what is RCX address and then set bp on user32!DispatchMessageW, then remove and find RCX+0x8 value. But in that case still don't does it what I want. But this demo4 shown somehting  what I will need and develop
 <br /><br />
TODO.......
