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
- In 3-4 places it is set to a fixed position, that's why it flashes so much - ```InvalidateRect(NULL, NULL, TRUE);``` - to get rid of the blinking change the parts of the code where this line appears
- CSS parsing only color, background-color, font-size
- HTML parsing only ```<p> <div> <<a href=\"> <a> <table> <tr> <td>``` tags
- JS parsing only this simple line of code ```strcpy(js.script, "console.log('Starting JS execution'); while(true);");``` in a simple way
- Color works in 2nd example from bottom. Take a look at this if you have to correct code in the current code at the top. From line ~909.

This demo does not use the NET library. I didn't intend to show anything complicated. This is supposed to be a simple introduction into an HTML, CSS and JS parser.

When you compile last example code from .cpp file then you will see the first demo version which has a simple CSS parser and 
```<p> <b> <i>``` tags

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/web_renderer/_web_renderer_example_from_cppfile_.png?raw=true)

But it's a long way from the look and functionality of EDGE. Javascript, JIT, JS debugger, how HTML structure is represented in the "Elements" tab, communication with the application via the console, e.g. writing document.body.remove(); to clear the canvas. There are many things to learn along the way to the goal of creating at least a semblance of the functionality that today's browser has. What does this type of application offer. Additionally, implementation of Video, GPU acceleration, porting to other devices such as ARM or Linux and Android systems. Today, the browser is a key element of the system and probably the one used the most by most people... Linus Torvalds  was right 12 years ago https://www.youtube.com/watch?v=ZPUk1yNVeEI&ab_channel=TFiR [Linus Torvalds: Why Linux Is Not Successful On Desktop] - "a lot of people do their work on basically in a web browser"

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/web_renderer/web_renderer_goal.png?raw=true)

But this is only small introduction to rendering a this simple code 
```
const char* html_content =
        "<div style=\"color:#FF0000;\">This is a div</div>"
        "<table>"
        "<tr><td>Row 1, Col 1</td><td>Row 1, Col 2</td></tr>"
        "<tr><td>Row 2, Col 1</td><td>Row 2, Col 2</td></tr>"
        "</table>"
        "<a href=\"http://example.com\">Click here for example</a>";
```
It would be nice to show a few more mechanisms here because it's a really interesting topic, but it's not the time for that now. 
<br />
Today's web browsers it's a cool thing. Nice tool. And now we go back to the C64 era, without this features and again "reinventing the wheel" with drawing this things using sprites hahahaha. The people behind the development of these tools are doing a good job.
<hr>
In this file https://github.com/KarolDuracz/scratchpad/blob/main/Win32/web_renderer/web_render_video.cpp there is an attempt to implement video rendering with directx 11. But there is a lot of bugs as you see on this small video below. There is some conflict between threads when rendering video and the JS parser. Maybe InvalidateRect is also messing something up. Maybe queue of messages https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-peekmessagea and this PM_NOREMOVE. Maybe timer. There is a Timer added 

line 805 ```case WM_TIMER``` 

 line 955 
```SetTimer(hwnd, 1, 16, NULL);``` 

and line 678 
```SetTimer(hwnd, 1, 16, NULL);``` 

Maybe is there conflict between timer and InvalidateRect and threds. Anyway. This demo was supposed to render random pixel noise as an image (video) on a page using DX11. Something is working. Not quite how I expect it to be yet, because it should display everything together and not like now, where even the RUN JS and STOP THREAD buttons are not visible because there is a conflict in rendering. Of course this is just a stupid demo. Performance is not important.

The line of background color is in 515 ```  float ClearColor[4] = { 0.0f, 0.125f, 0.3f, 1.0f }; // RGBA ``` when STOP THREAD button is pressed and "video" is rendering. This is the background area btw. Next time start first from here...

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/web_renderer/output_video_dx11.gif?raw=true)

Ok, there is some basis for further learning. For now, that's enough.
