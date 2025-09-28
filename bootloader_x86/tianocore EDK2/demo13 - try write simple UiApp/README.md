<h2>UiApp equivalent - simple demo</h2>
Default UiApp is here https://github.com/tianocore/edk2/tree/master/MdeModulePkg/Application/UiApp - But it has a structure, some API that can be used to 
scale it further. Current UEFIs already look great. it's not clunky anymore, it's just a cool design. But above all, it has to work. But there is another demo that tries to build something like this, just like UiApp does.

<h3>Demo 1</h3>

This is a much better implementation, but also very simple. It's a menu with three options. It's a simple animation of random pixels in different colors in the corner of the screen. But it works compared to Demo 2.
<br /><br />
Source code - https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo13%20-%20try%20write%20simple%20UiApp/demo1/HelloWorld.c <br />
Compiled demo1.efi - https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo13%20-%20try%20write%20simple%20UiApp/demo1/demo1.efi

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo13%20-%20try%20write%20simple%20UiApp/demo1_s1.png?raw=true)

<h3>Demo 2</h3>
Very poor quality code, many errors. Types mismatch. BUT IT WORKS. But probably only on EmulatorPkg and possibly QEMU. Obviously there is something wrong with the menu drawing.
<br /><br />
Compiled demo2.efi - https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo13%20-%20try%20write%20simple%20UiApp/demo2/gui_demo2.efi
<br /><br />
This is a continuation of the 10-12 demo. TimerLib is added to INF - https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo13%20-%20try%20write%20simple%20UiApp/demo2/HelloWorld.inf#L56 - This is needed for GetPerformanceCounter etc.
<br /><br />
I had to add something like this for the compiler because it was throwing an error that it couldn't resolve the dependency for _fltused - https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo13%20-%20try%20write%20simple%20UiApp/demo2/HelloWorld.c#L36
<br /><br />
In general, the code has several helpers that are THEORETICALLY supposed to correct such errors, but as you can see in the image, it still doesn't work properly. It needs to fix. Here is menu item 3, creating a small smoke at the bottom. <b>You can select from a list. Notepad shows the list of items on the left, and it also shows how to navigate using the UP/DOWN arrow keys in the GOP. You then need to press ENTER to enter that item, and then ESCAPE to exit the animation.</b><br /><br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo13%20-%20try%20write%20simple%20UiApp/demo2_s2.png?raw=true)

Here is a demo from Menu item 2, which is CREATURE, shows a simple animation.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo13%20-%20try%20write%20simple%20UiApp/demo2_s1.png?raw=true)

