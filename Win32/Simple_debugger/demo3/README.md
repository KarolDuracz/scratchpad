3-11-2024 - Back to this book (https://ksiegarnia.pwn.pl/Praktyczna-inzynieria-wsteczna,622427233,p.html?srsltid=AfmBOoooy1-iohVrjoNQa9FmNiLFVdlV_U9RGb8TYEa0cIq5CHVFa1hk) - Gynvael Coldwind, Mateusz Jurczyk - Praktyczna inżynieria wsteczna. There is a chapter written by hasherezade. And some demos uploaded to github as explanation in the code https://github.com/hasherezade/demos/blob/master/inject_shellcode/src/add_thread.h . <br /><br />

In this demo3 I want to take a closer look at CreateRemoteThread - https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-createremotethread
<br /><br />
In this add_thread.h from hasherezade github is there function run_shellcode_in_new_thread1 which demonstrate how run in some way CreateRemoteThread. And few other tips and tricks.
<br /><br />
1. Ok. For this particualr demo3 first we need create guest app.
```
// compile with /link user32.lib
// cl simple_demo.c /link User32.lib

#include <Windows.h>
#include <stdio.h>

int main()
{
	MessageBoxA(NULL, L"hello", L"test", MB_OK);
	while(1);
}
```
Compile it from commnad. But first we need setup environemnt for x64 bits. Following this guide (https://github.com/KarolDuracz/scratchpad/tree/main/Win32/Simple_debugger/demo2) or find in C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Visual Studio 2019\Visual Studio Tools\VC\x64 Native Tools Command Prompt for VS 2019.But we need setup environment for CL.exe for x64 bits in this case. Because host system and all processes is running as 64 bits.
```
 cl simple_demo.c /link User32.lib // compile

 cmd.exe > simple_demo.exe // and run
```
This app check if MessageBox is working and then jump to infinite loop.
<br /><br />
2. Ok, next go to main project (https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple_debugger/demo3/Project15/Project15/main.cpp) . This is quite a mishmash. But the uncommented code is what is currently to compile as a project and run. The rest is a few other tests of the same thing. But just compile this code as is and run it. Only change line 118 to process to debug. 
```
DWORD pid = GetProcessIdByName("notepad.exe");
```
I tested it on 2 64 bit processes. simple_demo.exe and notepad.exe. For notepad.exe put like this GetProcessIdByName("notepad.exe"); that's it. This find process using CreateToolhelp32Snapshot etc. and return handle to process and PID. Look at implemenation in .../Project15/main.cpp
<br /><br />
<h2>3. First run.</h2>
<br />
1. Compile Project15<br />
2. In CMD.exe jump to path. In my case "C:\Users\kdhome\source\repos\Project15\Debug\main_test" to simple_demo.exe<br />
3. Rus simple_demo.exe (cmd > simple_demo.exe)<br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple_debugger/demo3/pics/screen%201%20-%20run%20as%20admin%20simple_demo%20exe.png?raw=true)

4. Craete another CMD.exe but as Administrator <br />
5. Jump with this administrator CMD to "C:\Users\kdhome\source\repos\Project15\x64\Debug" - remember this is x64 so jump to \x64\Debug\<br />
6. In Project15 change line 118 . Compile it with CTRL + SHIFT + B . And run in CMD as administrator Project15.exe<br />
7. This create inside simple_demo.exe process new thread (in this example with TID 11704) - Right now is suspend.<br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple_debugger/demo3/pics/screen%202%20-%20create%20remote%20thread.png?raw=true)

8. Run WinDbg x64 as administrator<br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple_debugger/demo3/pics/3%20-run%20windbg.png?raw=true)

9. Press CTRL + C to stop Project15.exe<br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple_debugger/demo3/pics/4%20-%20CTRL%20V%20to%20stop%20project15%20exe.png?raw=true)

10. In windbg press "r" to check registers. But this is not necessary at this moment. Then find MessageBoxA using <br />

```
x user32!MessageBox*
```

and then set break point to address USER32!MessageBoxA 

```
 bp 00007ffd`2a908320
```
And press "g" to run again<br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple_debugger/demo3/pics/5%20-%20run%20g.png?raw=true)

11. Next you need to select process in ProcessExlporer and resume thread. 

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple_debugger/demo3/pics/6%20-%20cd.png?raw=true)

12. Then debugger hit in break point . Show registers using "r". Important register is RDX here.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple_debugger/demo3/pics/7%20-%20when%20hit%20bp%20look%20like%20that.png?raw=true)

13. Execute few instructions for example like that using "p" and "t" command in windbg. From MessageBoxA we jump to MessageBoxTimeoutA and then inside this will be call    MBToWCSEx

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple_debugger/demo3/pics/8%20-%20ida%201.png?raw=true)

14. Fast look to IDA and user32.dll

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple_debugger/demo3/pics/9%20-%20cd.png?raw=true)

15. Ok, press "r" in windbg and enter to run process. This execute thread and show MessageBox as you see on the screen bellow.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Simple_debugger/demo3/pics/10%20-%20press%20g%20and%20enter%20and%20get%20message%20box.png?raw=true)

<h3>For this issue we create Remote Thread inside other process and run step be step using WinDbg successfully.</h3>
<hr>