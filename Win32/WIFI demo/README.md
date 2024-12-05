<h2>TODO</h2>
Get live stream from GoPro cam. For android and apk XDV it works. But how to do this on windows? How to capture live streaming from camera for example to use in car as an auxiliary reversing camera.

```
const std::wstring ssid = L"X10000_01cf";
const std::wstring password = L"1234567890";
```

This is not topic to learn for now, but I will probably touch on this topic in a later phase regarding the network. 

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/WIFI%20demo/21%20-%205-12-2024%20-%20gopro%20cam.png?raw=true)

<h2>To fix</h2>
1 - This is not http from what wireshark shows, only SMB on port 217 and 218. UDP. <br />
2 - First I need to find way to get data from it. This sends something after catch connection. I see it in wireshark. ~300 packets. I don't know, maybe for this class device there is some protocol... ???
3 - Then proceed to display the image as a video stream. After fixed 1 and 2. example code (https://github.com/KarolDuracz/scratchpad/blob/main/Win32/WIFI%20demo/capture_frames_example.cpp)
