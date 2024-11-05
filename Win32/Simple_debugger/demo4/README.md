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
1. bp user32!DispatchMessageW // break point user32!DispatchMessageW
2. When you press in windbg "g" and move mouse on the window or restore from menu bar only debugger catch break point here. What is interesting to me is the MSG structure

```
// Run the message loop
    MSG msg = {};
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
```
example what looks like but in real x64 bit process this all have if I correctly think 8 bytes
```
user32!tagMSG
   +0x000 hwnd            : 0x12345678 HWND
   +0x008 message         : 0x201 (WM_LBUTTONDOWN)
   +0x010 wParam          : 0x1
   +0x018 lParam          : 0x20001
   +0x020 time            : 123456789
   +0x024 pt              : _POINT ( X = 0x100, Y = 0x200 )
```
3. dt user32!tagMSG @rcx   // 64-bit systems or dt user32!tagMSG poi(esp+4) // 32 bit system<br />
3a.  When "Symbol user32!tagMSG not found." you'll need - Typically nn 64-bit systems, the MSG pointer is typically found in the RCX register (for x64 calling conventions). On 32-bit systems, it will be on the stack (esp+4).<br />
4. ? rcx + 0x8 // identify the addres <br />
5.  ba r4 0x00007FF683590008 ".if (poi(0x00007FF683590008) == 201) { .echo Condition met: *(RCX + 0x8) == 201; }" // setup break point on address with contion RCX + 8 == 201 this is for Left mouse click on window <--- but for me this is not working properly<br /> 

<br /><br />
This must solve scenario when user KNOW what is  ```user32!DispatchMessageW```  address (but for all process is the same) and then set bp on user32!DispatchMessageW, then remove and find RCX+0x8 value which corespont to ```+0x008 message``` . But in that case still don't does it what I want. But this demo4 shown something  what I will need and develop. If you look at image you see there is somethings wrong with value from context.Rcx + 0xXX but it's ok for now. In the meantime everything will improve and come together as a whole.
 <br /><br />
TODO.......
