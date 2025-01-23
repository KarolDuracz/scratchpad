Update 23-01-2025 - 5:08 - In the "switch gop" folder (should be "switch gop", heh typos non-stop. Nevermind). In this folder, I uploaded a simple test to check if these registers from Intel HD 3000 will change something when I change the GOP mode from 0 to 1 and vice versa. In line 450 https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo4%20-%20VGA%20controller/swtich%20gop/HelloWorld.c#L450 the code in the long loop for x3 checks the value of tmpVal2 and state1. And every 0x1000 it resets tmpVal2 and then depending on the value of state1 it sets 1 or 0 and passes it to mGraphicsOutput->SetMode on line 457. Because there is an else if on line 452 so that after checking the condition and detecting ONLY one of them it exits the check and goes to the execution of the rest of the code. (...) OK. I don't have a picture, but I've uploaded a compiled HelloWorld.efi file. You can simply rename it to bootx64.efi and copy it to \EFI\Boot\ and run like the rest of the demo. <br /><br />
So back to the point. I wanted to see if the registers displaying would show different values ​​when I change the mode from 0 to 1 and vice versa. AND NOTHING CHANGES IN THE REGISTERS. The GOP mode changes, the screen actually shows that it's 800x600 60Hz, or 1024x768 60Hz, but the register values ​​are the same in both cases. For now timers don't work for me (I must figure out many topics like timers but...), so slow down the change in such a simple way by doing it in a loop. These values ​​are displayed on the screen in Y = 40 X = 10 and Y = 40 X = 11 for tmpVal2 and state1.<br /><br />
The purpose of this demo was to understand how high resolution (1600x900, 1366x786 etc) works when there is no access to the GOP, only a graphics card is detected on PCI and I have a BAR dump 0 - 5, with buffer addresses etc. In general, I have never tried to understand how displays work before. But now I want to know. That's all for now.

<h2>Quick dump of VGA controll registers Intel 3000</h2>
This is only dump from GOP and VGA registers from my machine. <br /><br />
I used documentations from link in my case - https://www.intel.com/content/www/us/en/docs/graphics-for-linux/developer-reference/1-0/intel-core-processor-2011.html - I have Intel i3 with integrated Intel HD 3000.
<br /><br />
Compile the same as previous demos.
<br /><br />
bootx64.efi << compiled current code for this demo4 . GOP + dump registers VGA from IO address 0x3CF, 0x3D4, 0x3D5, and registers GR10 etc.<br />
HelloWorld.c - code<br />
HelloWorld.inf - added only IoLib<br /><br />

This is exactly in sequence what you see on the right. See in the code. Before that there is GRX register with offset 0x5, 0x10, 0x11. Then 0x5f - HorizontalTotal, 0x4F - HorizontalDisplayEnd, and so on... Previously there is GOP information, as you can see only 2 modes. Currently mode 1, which is 320 x 258. I'm trying to figure this out on real hardware, rather than creating simple stupid virtual demos. I don't really know what I'm doing yet. But this result for this demo 4.<br /><br />
320 x 258 in hex, for decimal is equal to 800 x 600 <br />
400 x 300 hex = 1024 x 768 px
<br />
But right now I don't understand why this is only 80 x 32...
<br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo4%20-%20VGA%20controller/268%20-%2022-01-2025%20-%20zrzut%20z%20rejestrow%20VGA%20intel%203000.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo4%20-%20VGA%20controller/266%20-%2022-01-2025%20-%20bedzie%20lista%20rejestrow%20ale%20jeszcze%20ten%20potrzebuje.png?raw=true)

This is what a typical screenshot looks like for this example
```
Horizontal Timing:
  Total: 800
  Display End: 640
  Blank Start: 656
  Blank End: 752
  Sync Start: 656
  Sync End: 752

Vertical Timing:
  Total: 525
  Display End: 480
  Blank Start: 490
  Blank End: 491
  Sync Start: 490
  Sync End: 492

Other Settings:
  Interlaced Mode: No

Detected Resolution: 640x480
```

<br />
References:<br />
[ 1 ] https://www.virtualbox.org/browser/vbox/trunk/src/VBox/Devices/Graphics/ - Virtual Box source code 
