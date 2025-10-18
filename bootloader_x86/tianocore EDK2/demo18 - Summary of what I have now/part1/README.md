Safety first.
<br /><br />
In the previous "demo10 hello world extended" I showed how to install protocols. The main question is whether such a demo SHOULD BE RUN on real hardware? It's not recommended. I'm not sure if demo 10 directly interferes with the firmware (I don't know enough at this point to be certain that it could damage the firmware). But I'd rather not check. Therefore, this demo is strictly for EmulatorPkg or VirtualBox/QEMU. You can then reinstall the virtual machine there. The next demo that triggers the light is Demo 11 – Install ACPI table protocol in EmulatorPkg. This is definitely dangerous for real hardware, running it on real hardware can cause irreversible damage to the firmware. This already integrates the ACPI table, and it's likely impossible to restore the initial state after installation. That's why I haven't tested it on my own "real hardware," but it's here to test some of its capabilities, as you can test almost anything on the EmulatorPkg or virtual machines. For educational purposes. Furthermore, there are demos that use device registers ( sound demo ). These are currently emulated, but they use host devices, meaning tampering with registers and configuration can lead to irreversible damage.
<br /><br />
I wanted to write about this to remind myself, and to the reader who stumbles upon it, that I DO NOT RECOMMEND running this code as you wish. It can have irreversible consequences for the firmware and hardware. I realized how important this is when I hit a wall with USB analysis. I've reached a point where I currently don't know how to extract more information than I currently have. I'll present it further down in this post.
<br /><br />
There are many things that require very secure ( safe and health for device and firmware that exists inside now in laptop ) testing, and this can't be done with GPT-5 (?). Even if it suggests many solutions and provides guidance, errors do occur, and just as code requires analysis, what GPT-5 writes and the solutions it suggests require verification.
<br /><br />
My USB analysis stopped at this:<br /><br />
1. I checked a few simple diagnostic demos that extract some information about PCI devices and registers, etc., to perform host controller enumeration, such as OHCI or EHCI.<br /><br />
2. I learned from this that VirtualBox has two controllers: OHCI and EHCI. OHCI is ProgIf 0x10, and ehci is ProgIf 0x20. This way, you can tell which controller is which. My ASUS doesn't have OHCI, only two EHCI controllers. Two EHCI controllers, one with devices connected to it, such as a mouse, keyboard, and touchpad, and the other with Bluetooth and a built-in 2.0 Mpix camera.<br /><br />
3. At some point, I created a larger script, 4000 lines of code, that checked the USB ports more precisely to see if they were responding to EFI_USB_IO_PROTOCOL. I managed to read information from a USB flash drive (pendrive) such as INQUIRY: Vendor="Generic" Product="Flash Disk" Rev="8.07", DeviceDescriptor: Vendor=0x058F Product=0x6387 bDeviceClass=0x00. This device was detected in my previous demo because it responds to the basic EFI_USB_IO_PROTOCOL protocol.<br /><br />
4. So, since the pendrive responds, returns a descriptors, and returns a header with the config. I mean, I can read information directly from the USB flash drive's firmware, like its name as a string. This made me realize that the device is being detected correctly, so I'll try interacting with it using other mechanisms. Maybe I'll try writing something to it, BUT DOING IT VERY CAREFULLY AND SAFELY. So I created a demo that saves logs to the /EFI/Boot/myLogs/ path, and guess what? It doesn't work on QEMU. It worked on VirtualBox. It saved a log, and the entire test passed fine. So I checked it on ASUS (real hw) and it also saved a test log :) So, the conclusion and lesson learned were that the basic protocols and information I managed to extract showed that the device was INITIALIZED, DETECTED, ENUMERATED, AND CONFIGURED – READY TO WORK. And this is exactly what happens, since you can perform a file save operation directly to a specific path on this device. What's more, you can open a file, change it, and then delete it.<br /><br />
5. So the demo with logs saved to /EFI/Boot/myLogs/ showed me that I misunderstood what I was seeing, the demo I was running. That what I was seeing in this demo were devices that were detected and were ready to work. I could interact with them. But following this line of reasoning, since I see four devices on the list responding to EFI_USB_IO_PROTOCOL, I should be able to interact with them, but I don't. The other devices are a keyboard, mouse, and touchpad (probably by analyzing Windows' device manager and device trees). I tried testing the demo with VirtualBox and real hardware, but I couldn't find a device that matched the mouse class. Perhaps I should use Simple Pointer Protocol protocols directly here. That is, UEFI protocols, which are directly for keyboard or mouse (pointing devices). Maybe I'm approaching this topic incorrectly...?!<br /><br />
6. Moving on, I left the topic of the remaining devices aside, since I managed to find out that the demos I've made so far are unable to detect the mouse and download RAW packets, for example, for 5 seconds when I move the mouse. And the system still only sees one device, a USB flash drive -> [AsyncListen] Handle 0: VID=0x058F PID=0x6387 bDeviceClass=0x00, [AsyncListen] Handle 0: no Interrupt-IN endpoint (skip). So I started wondering, "Maybe I should check the NETWORK protocol." Maybe it works similarly—if PCI detects something, the USB controller does too. Maybe these ports aren't enabled but are detected, maybe it's still possible to detect something on them using NETWORK protocols. And to my surprise, in VirtualBox, it detected some devices and listed them. But in real hardware, it detected nothing! It became clear to me why the Bluetooth protocol didn't detect the device in the previous demo.<br /><br />
7. And here I come to the crux of the matter. That is, why some devices are detected and others aren't. And this is where I hit a wall. At this point, I'm not entirely sure what the next safe steps are that won't damage anything but will allow me to diagnose why these devices, despite being physically connected to the motherboard, are being detected in posts by NOT RESPONDING TO PROTOCOLS. GPT-5 has suggested a large number of possible causes. I'll present them below as a summary of the current state of affairs for clarity. And at the same time, to summarize this stage of learning. This could be a matter of tampering, which is dangerous because it attempts to perform DXE and PEI, a phase initiated by the firmware. (I DON'T KNOW, so I'll just stop here and present everything I have so far.)<br /><br />
8. So, I've ended up not knowing whether to interfere and continue running any code provided by GPT-5 because it could damage the original firmware. But on the other hand, since no NEWTORK protocol responds, how can I further diagnose that this device isn't INITIALIZED? Is it not running? Is there no driver? I don't understand this at this point. At this point, I also don't know what steps to take to diagnose this and even enable downloading packages when moving the mouse. There are a few things I'm stuck on right now. ABOVE ALL, IT MUST BE SAFE, without interfering with critical device registers or making changes that could damage the firmware.
<br /><br /><br /><br />
<h3>IMPORTANT:</h3>
I ​​DID NOT INCLUDE ANY PRE-COMPILED .efi FILES FOR SECURITY REASONS FOR ALL CODE IN THIS DEMO (summary). These codes relied on handler addresses, device IDs, etc., which may not be detected at all in other systems. While some handlers might overlap with this demo in VirtualBox, the USB flash drive I have, its vendor, and ID will not be included. Some parts of the code specifically searched for only the device or host it was connected to. Therefore, I'm only including code that can be traced and compared with the logs.
<br /><br />
Note: Some logs (text) have truncated lines or overlapping lines. This is due to PUTTY, which sometimes interrupts communication, making important sections of the log unreadable. However, they appear correctly in the terminal GOP. But for now, this log only serves as an outline of what is returned by the program or device in a given sequence/demo.

<h3>SUMMARY</h3>

This demo, with the saving to a USB flash drive under /EFI/Boot/myLogs/, also shows that it would be worthwhile to modify the shell to save what's printed on the console to a buffer somewhere, so it can then be saved to the logs. Instead of taking snapshots like we do now. This requires modifying the entire shell, but that's not what I want to focus on right now. That would be useful, but I want to focus more on small demos that test certain mechanisms, not on building a firmware that already works as a small system. Since we have SCSI, why not expand it beyond just saving logs to the myLogs folder? But I'm starting to wonder if we need to enable the MP protocol, timers, and other mechanisms that would allow us to collect these logs from running demos and then save them as a file. Maybe in the near future I'll try to expand this, but for now I have to stick with taking snapshots, even if they're not the best quality and difficult to read, because it's simply easier for me right now. But initially there is a test on real hw and it saved a successful test log, so it gives some outline of how it might work.

<h3>A few pictures and a short description of what will be discussed in parts 2 and 3</h3>

A photo from real hw confirming that the log is saved on a USB flash drive - everything is OK

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo18%20-%20Summary%20of%20what%20I%20have%20now/part1/images/1760750179758.jpg?raw=true)

Here you can see (I will put code and logs in part 2 and 3) that the code that detects devices on VirtualBox (next image) using these protocols, on my ASUS does not detect anything. Even though the logs show that the devices are connected to the controller ports. And that got me thinking.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo18%20-%20Summary%20of%20what%20I%20have%20now/part1/images/1760790821841.jpg?raw=true)

VirtualBox does not detect only a few, which is expected looking at the pictures below and which ETHERNET devices are visible after Windows 10

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo18%20-%20Summary%20of%20what%20I%20have%20now/part1/images/248%20-%2018-10-2025%20-%20trzeba%20przerobic%20shell.png?raw=true)

Devices on Win 8.1 host

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo18%20-%20Summary%20of%20what%20I%20have%20now/part1/images/249%20-%2018-10-2025%20-%20host%20win81.png?raw=true)

Detected network adapters on VirtualBox 

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo18%20-%20Summary%20of%20what%20I%20have%20now/part1/images/252%20-%2018-10-2025%20-%20%20nie%20bedzie%20wifi.png?raw=true)

I'll come back to this: save logs to /EFI/Boot/myLogs. This doesn't work in QEMU. However, Qemu behaves differently than VirtualBox. Qemu doesn't disconnect the USB flash drive from the host system, and I still see the tray icon at the bottom right. VirtualBox reconnects the device, and I can't see it or use it through the host. Windows 8.1 is connected to VirtualBox. That's why QEMU returns the error here, but this is a preliminary cause and diagnosis.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo18%20-%20Summary%20of%20what%20I%20have%20now/part1/images/243%20-%2018-10-2025%20-%20cd.png?raw=true)


<h3>A few low-quality photos from a USB test that tries to detect pointing devices like mice, this class of devices, and download the RAW data packet it was sending. FAILED</h3>

The code and explanation will be in part 2 and 3

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo18%20-%20Summary%20of%20what%20I%20have%20now/part1/images/test%20realhw1.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo18%20-%20Summary%20of%20what%20I%20have%20now/part1/images/2.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo18%20-%20Summary%20of%20what%20I%20have%20now/part1/images/3.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo18%20-%20Summary%20of%20what%20I%20have%20now/part1/images/4.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo18%20-%20Summary%20of%20what%20I%20have%20now/part1/images/5.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo18%20-%20Summary%20of%20what%20I%20have%20now/part1/images/realhw_test6.png?raw=true)



