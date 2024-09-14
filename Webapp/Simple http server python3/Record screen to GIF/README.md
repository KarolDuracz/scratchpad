![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Webapp/Simple%20http%20server%20python3/Record%20screen%20to%20GIF/output.gif?raw=true)

<br />
Ok, what's that? <br />
There is a lot of stuff about recording screen etc. But this solve the my problem - how to capture screen with 30 fps for 10s and save to GIF file. <br />
<br />
First we need some dependiences. This is list of current version which I installed to run this script here.<br />
1. mms 9.0.2 <br />
2. imageio 2.35.1 <br />
3. pillow 10.3.0 <br />
4.  1.26.4 <br />
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
