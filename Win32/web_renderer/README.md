![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/web_renderer/output_web_renderer.gif?raw=true)

DEMO EXPLAINED <br />
The main idea is to create a simple HTML parser and a simple JS parser to a graphical form using GDI or DirectX. And add a simple debugger to execute JS code. As in the previous examples, there are several implementations in the .cpp file. But what you see in the picture is the current one. This is just a very simple version to get some insight into the subject more deeply. Simple introduction to start somewhere. 

On this .gif you see fast demo, but:
1. first you can see that I click the RUN JS button 3 times. If you look at the implementation you will see that it creates an InfiniteLoopThread. You can see in the background on PE that there are 3 threads in this process. This is why counter is counting so strangely because there are 3 threads updating the value of "counter" and they have a copy of the value it looks like here.
2. Next, I clicked on STOP THREAD button.
3. Then I clicked on the link "click here for example" on the right top sight page (link is created in CSS styles. See implementation in CSS parser code)
4. Next I clicked F5 few times to refresh page, as you see on the console.
5. Then I quickly clicked RUN JS, STOPPED and again RUN JS and STOPPED.  And on the link. And then F5 key to refresh.
6. And finally I clicked RUN JS again which created 1 thread which is put to sleep for 100 ms and at the end of recording this video I clicked STOP. End of recording.

TODO
- In 3-4 places it is set to a fixed position, that's why it flashes so much - InvalidateRect(NULL, NULL, TRUE); - to get rid of the blinking change the parts of the code where this line appears
- ...