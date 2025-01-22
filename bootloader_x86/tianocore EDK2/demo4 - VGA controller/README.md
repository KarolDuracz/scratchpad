<h2>Quick dump of VGA controll registers Intel 3000</h2>
This is only dump from GOP and VGA registers for my machine.<br /><br />
This this get documentations from link in my case - https://www.intel.com/content/www/us/en/docs/graphics-for-linux/developer-reference/1-0/intel-core-processor-2011.html - I have Intel i3 with integrated Intel 3000.
<br /><br />
Compile the as previous demos.
<br /><br />
bootx64.efi << compiled current code for this demo4 . GOP + dump registers VGA from IO address 0x3CF, 0x3D4, 0x3D5, and registers GR10 etc.<br />
HelloWorld.c - code<br />
HelloWorld.inf - added only IoLib<br /><br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo4%20-%20VGA%20controller/268%20-%2022-01-2025%20-%20zrzut%20z%20rejestrow%20VGA%20intel%203000.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo4%20-%20VGA%20controller/266%20-%2022-01-2025%20-%20bedzie%20lista%20rejestrow%20ale%20jeszcze%20ten%20potrzebuje.png?raw=true)

<br />
References:<br />
[ 1 ] https://www.virtualbox.org/browser/vbox/trunk/src/VBox/Devices - Virtual Box source code 
