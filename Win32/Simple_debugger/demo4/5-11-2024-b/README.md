NOT FIXED <br />
you must change<br />
In debugger.cpp in line 97 : char ret[64]; to >>>> unsigned char ret[64]; <br />
This means, form type ```char``` to ```unsigned char``` !
<br /><br />
And you must change line 99 in debugger.cpp to
```if (ReadProcessMemory(hProcess, (LPCVOID)context.Rcx, &ret, 64, &ret_bytes)) {	```
fix that value context.Rcx.<br />
And then this look OK. From 0x8 offset is 0x200 and this is correspond to ```WM_MOUSEMOVE``` 
```
// C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\um\WinUser.h
#define MN_GETHMENU                     0x01E1

#define WM_MOUSEFIRST                   0x0200
#define WM_MOUSEMOVE                    0x0200
#define WM_LBUTTONDOWN                  0x0201
#define WM_LBUTTONUP                    0x0202
```
So, this is looks correct. But for this case I need to filter each type of events and message. But this is yet to be done here. To be sure that this is correct you would need to check HANDLE window from first parameter, first 0-7 bytes. But... not this time.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple_debugger/demo4/5-11-2024-b/82%20-%205-11-2024%20-%20message%20is%20comming%20looks%20correct.png?raw=true)

<br /><br />
<b>fixed</b>
<br /><br />
1. Add PID as window title in SimpleGUI.cpp ```SetWindowTextA(hwnd, std::to_string(GetProcessId(GetCurrentProcess())).c_str());``` <br >
2. Add some stuff to debugger.cpp but still not working well.
<hr>
<br />
For this case we can use the Spy tool with

```
C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\Tools\spyxx_amd64.exe
```

And now I quickly checked if the address from the first 0-7 bytes is correct. And it is ok.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple_debugger/demo4/5-11-2024-b/use%20Spyx64%20to%20get%20info%20about%20handles.png?raw=true)
