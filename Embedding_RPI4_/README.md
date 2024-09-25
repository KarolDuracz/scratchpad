![dump](https://raw.githubusercontent.com/KarolDuracz/scratchpad/refs/heads/main/Embedding_RPI4_/rpi%204%20and%20nucleo.png)

I bought it some time ago RPI4 with 2 GB ram and camera. After reading this course. <br />
https://forbot.pl/blog/kurs-raspberry-pi-od-podstaw-wstep-spis-tresci-id23139
 <br /> <br />
https://botland.com.pl/moduly-i-zestawy-raspberry-pi-4b/14646-raspberry-pi-4-model-b-wifi-dualband-bluetooth-2gb-ram-18ghz-765756931175.html?cd=1050025856&ad=51004438223&kd=&gad_source=1&gclid=EAIaIQobChMI8ovi_6vdiAMVzJSDBx0moyWLEAAYAiAAEgKg__D_BwE <br />
Opis produktu: Raspberry Pi 4 model B WiFi 2x microHDMI, USB C, USB 3.0, 2GB RAM 1,8GHz - Broadcom BCM2711 quad-core 64-bitowy ARM-8 Cortex-A72 1,8 GHz <br />

https://botland.com.pl/kamery-do-raspberry-pi/6124-raspberry-pi-camera-hd-v2-8mpx-oryginalna-kamera-do-raspberry-pi-652508442112.html <br />
<hr>
First, I buought model from this family processors and boards (NUCLEO-L031K6) - not exactly this model but from Rafał's tutorial. But this board on picture looks like model which I have.
https://www.tme.eu/pl/details/nucleo-l031k6/zestawy-do-ukladow-stm/stmicroelectronics/?brutto=1&currency=PLN&utm_source=google&utm_medium=cpc&utm_campaign=P%C3%B3%C5%82przewodniki%20PL%20[PLA]%20CSS&gad_source=1&gclid=EAIaIQobChMIz8CU563diAMVpUBBAh1fZwG6EAQYAiABEgII-vD_BwE <br />
And for this tiny device is introduction to programming in magazine NO 73, 74, 75 - very qood lecture btw. from Rafał Kozik.<br />
https://programistamag.pl/programista-6-2018-73/  <br />
<br />
Of course I have an Arduino Uno also.<br />
<br />
I worked for a year in a company in the automotive industry. They dealing with the production of phones for cars and satellite navigation, maps, GPS etc. Most of the boards had Cortex M3/M4 MCUs and Texas Instruments Delfino C2000 if I remember correctly. 
But definitely the Delfino family of processors with 166 MHz clock up to 200-320 I think. For GPS and GSM transmission they used UBLOX modules. And that's what got me interested in buying these boards and learning more about how to program them. And since then I've learned the basics, but that's all. I did these tutorials 
from forbot.pl to install Rasbian, start the camera and record something CV2, motion detection. And on Nucleo hello world, i.e. diode blinking, fibonacci to demonstrated stack opperations. And that's it. 
I worked on SMT line, but this was interesting experiences. Thanks to this, you will learn about the structure of the device, components, etc., as well as the code that is loaded into the MCU memory.
<br /><br />
<b>But what's interesting for me at this moment is to move some stuff from Machine Learning to this small devices with ARM Cortex A72. Because today's mobile devices use for example MediaTek Helio G35 with Cortex A53.
And I want to move in future to learn something about Android and programming mobile devices. Write a simple application using ML and a camera for image analysis etc. But this is not goal to 2025, even 2026. But keep in mind.</b>
<hr>
For experiments purpose with linux kernel, SPI bootloader etc better is to buy raspberry pi zero, probably this model (without W - no bluetooth, no wifi) https://www.raspberrypi.com/products/raspberry-pi-zero/ || https://www.tme.eu/pl/details/sc0020/raspberry-pi-minikomputery/raspberry-pi/raspberry-pi-zero-w/?brutto=1&currency=PLN&utm_source=google&utm_medium=cpc&utm_campaign=P%C3%B3%C5%82przewodniki%20PL%20[PLA]%20CSS&gad_source=1&gclid=EAIaIQobChMIguXG6NbdiAMVm4poCR2BsxoAEAQYAyABEgKSRfD_BwE <br />
And again, Rafał Kozik did awesome work writing this article about bootloader on RPI ZERO https://programistamag.pl/programista-52019-84-wrzesien-2019-wlasny-linux-na-raspberry-pi/ <br />
Right know I don't know for 100% is PICO enough for these experiments but it's good to know that there is an even cheaper board  https://www.tme.eu/pl/details/sc0915/raspberry-pi-minikomputery/raspberry-pi/raspberry-pi-pico/?brutto=1&currency=PLN&utm_source=google&utm_medium=cpc&utm_campaign=P%C3%B3%C5%82przewodniki%20PL%20[PLA]%20CSS&gad_source=1&gclid=EAIaIQobChMIguXG6NbdiAMVm4poCR2BsxoAEAQYASABEgKQ-fD_BwE
But it's probably better to take the version with BT and WIFI since the price is the same shelf to have more to interact with the equipment. Today the price of RPI ZERO is ~70 PLN, and PICO is ~20 PLN. But probably for pico you have to use freeRTOS. But for me better is to implement this as exercise https://github.com/rosly/rados | https://programistamag.pl/programista-112015-42-piszemy-wlasny-rtos/
<br /><br />
BUT THIS IS NOT IMPORTANT TODAY. <b>This is for me, reminder, my learning path.</b>
<hr>
But that was a few years back. Today there are a lot of other interesting boards that use FPGA systems, but I'll probably start with https://rysino.gitlab.io/pl/ if I come back to these topics. Maybe I will manage to 2026 - 2027 :)
<br /><br />
Around this whole EMBEDDING term there is a lot of cool stuff. Not only blinking leds and load linux kernel to type some characters via USB. But today my "skills" stopped at running these hello world and that was the end of my learning. 
<hr>
In this tutorial they used raspistill (https://forbot.pl/blog/kurs-raspberry-pi-kamera-zdjecia-transmisja-id24480) but for me this
is not working. I was looking how to fix that. I don't remember but the problem is hardware revision and some compatibility either 32 bit raspbian or some driver in this revision of the camera. I don't remember now, but there is something to this model RPI or camera. But libcamera library works fine. And motion detect also. Below you don't see picture from this little camera, but this show part of RPI desktop screen with libcamera short test in console as you see on geany in line 2. This is only as reminder for me. Maybe I'd go back here 2-3 years, maybe further maybe not. But the Linux kernel still has many secrets for me to discover so I'll probably come back here soon.
<br /><br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Embedding_RPI4_/1727259573848.jpg?raw=true)
