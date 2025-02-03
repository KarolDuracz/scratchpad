<h2>TODO</h2>
I'm starting this topic today as an "anchor" for me. From what I see, BEEP is the "simplest" because it is one of the smallest drivers/services
to analyze because it is only 8 kB - C:\Windows\System32\drivers\beep.sys. But at the same time it has interaction with the sound card process.
So this is quite interesting to analyze. From what I initially checked at some point beep.sys executes NtCraeteFile and NtDeviceIoControlFile 
from KernelBase.dll (from WOW64 folder if it is x86 process). And the second function contains the InputBuffer parameter which has as first 8 bytes 
the frequency and wavelength, that is the arguments Beep(1000, 450) as hexadecimal in stack. And you can change it using the debugger before 
executing the next instructions inside to generate a different type of sound.
<br /><br />
This is good exercises,before analyzing drivers for Intel HD 3000 and Nvidia GT540M. Because for my purposes I will need it.
<b>I plan to do it in the first half of 2025</b>. Analyzing the entire driver is not really an option, but what controls it after BasicDisplay. Generally, 
the BasicDisplay itself from Windows also requires analysis on the side.
<br /><br />
https://www.nvidia.com/en-us/geforce/drivers/results/103988/ - Windows 8.1 - this is driver for list of graphics cards<br />
https://www.nvidia.com/en-us/geforce/drivers/results/87791/ - Windows 10<br />
https://www.intel.com/content/www/us/en/download/17608/intel-graphics-driver-for-windows-15-28.html - Intel HD 3000 (win64_152824)
<br /><br />
But there are still some things to do along the way, even if it's just understanding the basic operation and configuration of the card.
Because BasicDisplay and BasicRender have support probably from GOP (uefi). But I'm not sure. But from what I see from Windows PE, 
it uses the maximum mode that GOP detects I think. And this is the base I think for BasicDisplay.sys. But after installation recommended 
drivers for particular GPU model, we have full access to high resolution and smooth graphics...
<br /><br />
For VM case it's important. There are several things here like fake device in system which can be detected by this driver. 
Like bridge between VM and Host to accelerate VM itself etc. And that's why I need to understand the basics of how it works.
<br /><br />
<h3>Just to be clear about what I'm doing here.</h3>
  Today, Nvidia releases the GeForce RTX 5090 Mobile for laptops. But from what I see on YT, some people are upgrading their laptops using the RTX 3090 and 4090. But the whole point of this exercise is to understand the driver in general. Because as you can see from the list of cards that this NVIDIA driver supports, there is a huge gap between my GT540M and the "newer" '14 system, the GTX 980M. To be clarify mainly mean the comparison from the "Render Config" table, although memory access time and memory are also important. GT540M has 2 SM and 96 units, 980M has 12 SM and 1536 units. But it's still the same driver, it's just looking for a different DEVICE ID on the list on INF file and dependiences for it.<br />
https://www.techpowerup.com/gpu-specs/geforce-gt-540m.c702 <br />
https://www.techpowerup.com/gpu-specs/geforce-gtx-980m.c2746 <br />
https://www.techpowerup.com/gpu-specs/geforce-rtx-4090-mobile.c3949 <br />
https://www.techpowerup.com/gpu-specs/geforce-rtx-5090-mobile.c4235 <br />
But you have to start somewhere, and my current chip to do exercises is this GT540M... from 2011. <br />
(I tried to understand something about my GT540M here https://github.com/KarolDuracz/scratchpad/tree/main/OpenCL%20via%20ASUS%20with%20GT540M)
<br /><br />
Try to do in first half of '25.
