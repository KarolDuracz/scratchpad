> [!WARNING]
> I DO NOT RECOMMEND RUN THIS ON REAL HARDWARE. I AM NOT RESPONSIBLE FOR THE CODE AND CONSEQUENCE.

I also tested these demos using Real HW on my Asus laptop. I'm writing this because after running demo, which attempts to initialize controller after a soft reset—simply resetting computer after entering Windows, was unable to initialize ETHERNET connection. It had this problem and repeated process again and again, looking at the behavior of the tray icon on right bottom corner, which tried to establish a connection but failed. Only shutting down computer fixed this error. This means the system and my firmware, which is currently executing in DXE phase of ASUS system, are probably doing something different than the sequence I've demonstrated here. But this is just a preliminary analysis.
<br /><br />
This is a screenshot of the execution of this SOFT INITIALIZATION which you can see below how it works on VirtualBox on the screenshot. <b>In short, program did not work as shown on virtual box.</b>

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo16%20-%20OHCI%20-%20first%20attempt/images/1760276157194.jpg?raw=true)


