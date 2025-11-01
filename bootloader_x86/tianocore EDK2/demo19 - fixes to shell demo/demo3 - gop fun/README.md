> [!IMPORTANT]
> The images aren't uploaded to github here to repo. I found them on Google. They weren't licensed. But I wanted to see what it looked like in REAL HW, how those pixels would be rendered. The images are really cool. I don't intend to steal anyone's work. I didn't ask the authors if I could. I was just looking for something Halloween-themed and to see what you looked like in that demo. And these seemed cool for that demo. Nothing more. THAT'S WHY THERE ARE NO IMAGES IN THIS REPOSITORY, THERE ARE ONLY SCREENSHOTS.

This demo isn't configured with the demo shell. It's just for playing with GOP a bit more. Fonts. How transparency works, etc. There is also a SIMPLE POINTER there https://uefi.org/specs/UEFI/2.10/12_Protocols_Console_Support.html#simple-pointer-protocol .But I've only tested it successfully on REAL HW. On AUSU, after entering this .efi demo and issuing the ```gop pointer_mode device``` command, holding the left or right mouse button and moving the mouse, you get logs on the console from the mouse position. After enabling ```gop pointer_mode on``` , you can see the mouse cursor, and if you hold and move the mouse button the same way, you can see logs from the position
<br /><br />
Files are loaded (HARD CODE) from the path - #define MYIMAGES_REL_PATH L"\\EFI\\Boot\\myPics"

<h3>Example commands</h3>

https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo3%20-%20gop%20fun/HelloWorld.c#L992

```
  Print(L"Simple interactive demo (examples):\n");
  Print(L"  gop list\n");
  Print(L"  gop set 1\n");
  Print(L"  gop bg FF00FF\n");
  Print(L"  gop fontbg transparent\n");
  Print(L"  gop fontsize 3\n");
  Print(L"  gop fontfg color FFFF00\n");
  Print(L"  gop draw 50 50 \"HELLO, WORLD!\"\n");
  Print(L"  gop blendtest\n");
  Print(L"  gop pointer_mode on\n");
  Print(L"  gop pointer_mode device\n");
  Print(L"  gop images list\n");
  Print(L"  gop images load 0\n");
  Print(L"  gop add_image 100 100\n\n");
```

Command that loads a custom font but only for the HELLO WORLD text at any position on the screen ```gop draw 500 400 hello``` . You can add several on screen on different location.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo3%20-%20gop%20fun/images/1762021892027.jpg?raw=true)

Alpha channel test ```gop blendtest``` ( transparency ) and loading images.
<br /><br />
```gop images list``` - it is not shown in the image here, but in the code you can check that this command is there and lists all BMP files from the \EFI\Boot\myPics\ folder. This also indexes files, so you don't load them by name but from the index.
<br /><br />
```gop images load 0``` - loading index 0 ( image that has been assigned index 0 )
<br /><br />
```gop add_image X Y``` - add_image and location X Y like you see in the console
<br /><br />
```gop``` - short list of commands
<br /><br />
```gop blendtest``` - creates these 2 rectangles that are transparent

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo3%20-%20gop%20fun/images/1762021892091.jpg?raw=true)

Loading another image from \EFI\Boot\myPics\ - to see a more complex and dynamic image as it is rendered

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo3%20-%20gop%20fun/images/1762021892104.jpg?raw=true)

Loading another image from \EFI\Boot\myPics\ - something bigger

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo3%20-%20gop%20fun/images/1762021892120.jpg?raw=true)

