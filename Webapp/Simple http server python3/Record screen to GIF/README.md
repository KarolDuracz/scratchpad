![dump](![image](https://github.com/user-attachments/assets/45313d8c-49ae-4f21-8a8e-fbac141274ee)

<br />
Ok, what's that? <br />
There is a lot of stuff about recording screen etc. But this solve the my problem - how to capture screen with 30 fps for 10s and save to GIF file. <br />
<br />
First we need some dependiences. This is list of current version which I installed to run this script here.<br />
1. mms 9.0.2 <br />
2. imageio 2.35.1 <br />
3. pillow 10.3.0 <br />
4. numpy 1.26.4 <br />
This is from command <br />

```
cmd.exe > python -m pip list | findstr numpy \\ imageio, pillow, mms
```
<br />
<b>RUN </b> <br />

```
cmd > python rec.py
```
<br />
- But for 10 seconds of recording this image, as you can see, it takes almost 10 MB for the .gif file! <br />
- In my old laptop this takes around 30s to generate and save this file. It takes a while. <br />
- Looking at ProcessExplorer and memory usage, it needed 3.2 GB of memory to generate this particular animation!!! A lot.
<br />
<br />
TODO : this script can't record extended display in windows. 

<br />
I do test for 30 sec with 30 fps: (in windows 8.1 with 8 GB RAM) <br />
To generate and saving file total time : 4.56 minutes<br />
record area: 1600 x 900 px <br />
cpu usage: mean ~15-20%<br />
private bates: max 9.5 GB !!! A lot.<br />
working set: max ~5 GB. mean 4.5 GB. <br />
output file size: ~40 MB.
<br />
Runing on IE V.11.0.9600.20671 after 15 seconds when private memory rise to ~1.1 GB and working set to ~1.4 GB IE stopped rendering this file ;p <br />
EDGE  109.0.1518.140 64 bit need less reources, ~66k MB private bytes and ~610k MB working set. And works fine. 
<br />
<br />
So. For 10 sec record is ok. But longer GIFS is not recommended to record with this "tool" :)
