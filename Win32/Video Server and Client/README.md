This not works. But this is base for develop in some time.
<br /><br />
<b>DEMO 1# - a few words of explanation</b><br /><br />
update - 05-09-2024 - Maybe after year I forget what is going on here :) A lot of code is to fix. But in file <br />
```
scratchpad/Win32/Video Server and Client/server_bottleneck/Untitled.ipynb
```
I can't check it 100% now, but looking at the source code "Win32/Video Server and Client/win_server_bottlenecks/win_server_bottlenecks/main.cpp" string "Welcome to the server" which is the server's response to the client is in line 888 !!! So, this is probably client for this last example from this main.cpp file. And this test was intended to create the maximum number of handles and connections supported by Windows and this server. Therefore 100 connections are created in this client file.
<br /><br />
<b>DEMO #2</b> <br /><br />
```
Win32/Video Server and Client/server_bottleneck/Untitled1.ipynb
```
This is also for one of the sample implementations of this server, but I don't remember exactly which one now. But the purpose of this was to create a website service for displaying videos similar to YT. Something like that. That was the purpose of this exercise and making this server but also to check the bottlenecks of the windows system and my ASUS laptop. That is, measuring the speed of writing to memory, to physical disk. Measure the overall throughput of the entire system in the task of streaming video via web app. But this stuff not working right now. There is only example how to do this maybe...
