<h2>TODO</h2>
Get live stream from GoPro cam. For android and apk XDV it works. But how to do this on windows? How to capture live streaming from camera for example to use in car as an auxiliary reversing camera. Camera using in this demo: Pro4U BLOW 4k, but the older version without front display, only with rear display. Camera has a WIFI mode after turn it on transmits video. And you can control basic functions from android app.

```
const std::wstring ssid = L"X10000_01cf";
const std::wstring password = L"1234567890";
```

This is not topic to learn for now, but I will probably touch on this topic in a later phase regarding the network. 

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/WIFI%20demo/21%20-%205-12-2024%20-%20gopro%20cam.png?raw=true)

<h2>To fix</h2>
1 - This is not http from what wireshark shows, only SMB on port 217 and 218. UDP. <br />
2 - First I need to find way to get data from it. This sends something after catch connection. I see it in wireshark. ~300 packets. I don't know, maybe for this class device there is some protocol... ??? <br />
3 - Then proceed to display the image as a video stream. After fixed 1 and 2. example code (https://github.com/KarolDuracz/scratchpad/blob/main/Win32/WIFI%20demo/capture_frames_example.cpp)
<hr>
This was only quick demo. But this is importants facts. XDV 1.9.62 version. And this app can manage 3 modes, and record stream. But when I switch to 720P 90/120FPS and swithc to 1080P 60FPS without recording, only set on menu this option, then the screen of stream becomes dark. As if the camera had lost focus and light. Event switch to 4K mode this not back to default settings of CMOS. Only turning on 4K recording resets these errors and the image becomes clearer and sharper, adjusting itself frame by frame in relation to lighting. I'm writing this because maybe there really is some protocol for this, but I don't know anything specific about it now.
  
