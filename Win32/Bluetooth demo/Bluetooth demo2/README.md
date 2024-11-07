Created 7-11-2024 - One more thing
<br /><br />
This is more deep topic than this demo. But what is interesting for me is recording voice commands and recognizing them to control the system, e.g. moving the mouse or typing on the keyboard while controlling it with voice (speech recognition). Specifically, through a very simple device F9 headsets. This device communicates via Bluetooth, can record, has touch control and a few simple commands that can be activated by touching the housing with finger (touch multi-function). This device has a range of ~10 meters and the battery lasts ~4 hours. 
<br /><br />
It may be responsible for recognizing commands and controlling the system WHISPER (https://github.com/openai/whisper/tree/main) or (https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition). To control with high precision you can train a network like nanoGPT with a different token structure. But for now it's theory. Practice will come soon. 
<br /><br />

https://manuals.plus/qf-tech/f9-bluetooth-earbuds-manual
<br /><br />

This example of recording sound for a few seconds to the file "C\Windows\Temp\out.wav" works, but the sound is noise, there is no clear speech. Either I have something set incorrectly in F9 or in the system. But the trick is to set the default device in the system and with this demo you can record commands lasting a few seconds.
<br /><br />

But this is not working properly right now. TODO.
<br /><br />

Windows has a mechanism for recognizing commands but I'm trying to approach the API manually ===> Control Panel\All Control Panel Items\Speech Recognition even win8.1

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Bluetooth%20demo/Bluetooth%20demo2/pic%201-%20demo%20bluetooth%20record%20audio.png?raw=true)

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/Bluetooth%20demo/Bluetooth%20demo2/pic%202%20-%20demo%20record%20audio.png?raw=true)

<hr>
There will be more demos here including a sniffer with RPI4 and wireshark to decode bluetooth packets and listen for low level communication. And looking deeper into Windows drivers. But this is more advanced topics for me. But I want to have this demo here.
