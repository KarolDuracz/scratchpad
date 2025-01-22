<h2>Quick dump of VGA controll registers Intel 3000</h2>
This is only dump from GOP and VGA registers for my machine.<br /><br />
This this get documentations from link in my case - https://www.intel.com/content/www/us/en/docs/graphics-for-linux/developer-reference/1-0/intel-core-processor-2011.html - I have Intel i3 with integrated Intel 3000.
<br /><br />
Compile the as previous demos.
<br /><br />
bootx64.efi << compiled current code for this demo4 . GOP + dump registers VGA from IO address 0x3CF, 0x3D4, 0x3D5, and registers GR10 etc.<br />
HelloWorld.c - code<br />
HelloWorld.inf - added only IoLib<br /><br />

This is exactly in sequence what you see on the right. See in the code. Before that there is GRX register with offset 0x5, 0x10, 0x11. Then 0x5f - HorizontalTotal, 0x4F - HorizontalDisplayEnd, and so on... Previously there is GOP information, as you can see only 2 modes. Currently mode 1, which is 320 x 258. I'm trying to figure this out on real hardware, rather than creating simple stupid virtual demos. I don't really know what I'm doing yet. But this result for this demo 4.

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
