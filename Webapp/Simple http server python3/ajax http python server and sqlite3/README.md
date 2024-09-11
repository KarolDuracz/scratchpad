AJAX + python http serve + sqlite3 - tiny example
<br />
1. How to run<br />
<b>cmd.exe > python server2.py</b><br /> - but this .db file from this repo is not needed, because python script create own .db file if not exists.
2. Like on this picutre. Open 2 pages in web browser. Open http://localhost:8080. To open logs http://localhost:8080/logs<br />
3. Click on the button "send signal" <br />

This might be use to save those tasks (collect data)  for example from this app --> https://github.com/KarolDuracz/scratchpad/tree/main/Webapp/Simple%20http%20server%20python3/pomodoro-app <br />
One thing that might be useful for this application is that after clicking "start" it will save the information to the server and sqlite3 database on the server side. And user can list all logs call up /logs in adrress bar.
<br />

![dump](https://raw.githubusercontent.com/KarolDuracz/scratchpad/main/Webapp/Simple%20http%20server%20python3/ajax%20http%20python%20server%20and%20sqlite3/40%20-%2011-09-2024%20-%20ajax%20http%20python%20server%20and%20sqlite3.png)

<br />
btw. <br />
This is for remind me, that in MSVC package is useful tool spyxx in folder "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\Tools\" . For this case this not very useful, but when you press CTRL + F in this tool opened "Find window" to select target window for "listening messages" like you see one this picture.
<br /><br />

![dump](https://raw.githubusercontent.com/KarolDuracz/scratchpad/main/Webapp/Simple%20http%20server%20python3/ajax%20http%20python%20server%20and%20sqlite3/41%20-%2011-09-2024%20-%20sec%20example%20with%20spy%20window.png)

<br />
since I'm already touching on network topics, here's a screenshot from wireshark. Similar to this tool is TCPDUMP https://www.tcpdump.org/manpages/tcpdump.1.html
<br /><br />

![dump](https://raw.githubusercontent.com/KarolDuracz/scratchpad/main/Webapp/Simple%20http%20server%20python3/ajax%20http%20python%20server%20and%20sqlite3/43%20-%2011-09-2024%20-%20sample%20from%20wireshark%20heh.png)

<br /><br />
-- also interesting topic - distributed computing -- <br />
https://github.com/gynvael/zrozumiec-programowanie/blob/master/021-Czesc_V-Rozdzial_14-Komunikacja_miedzyprocesowa/calc_server.c
<br />
there are a lot of interesting issues related to https connection and security also, but this is not repo and topic for this. This is for simple webapp to send some data to python http.server and write into .db via using sqlite3. But I remember this exercise from the book "zrozumieć programowanie - gynvael" is very interesting. <br />
So, for future learning I want to put here...
