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
<hr>
Actually, this topic should have started with rendering a simple HTML skeleton like this. And inside the rendering engine create an object to hold this page construction like those meta tags. <br /><br />

```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>html parser demo</title>
    <style>
        body { margin: 0; }
        canvas { display: block; }
    </style>
</head>
<body>

<script>
    console.log(1);
</script>

</body>
</html>
```

<hr>
Back to this [scratchpad/Win32
/Direct3D11 and Intel3000 or NvidiaGT540M] https://github.com/KarolDuracz/scratchpad/tree/main/Win32/Direct3D11%20and%20Intel3000%20or%20NvidiaGT540M 
When I started looking into what was going on there, it turned out that something was working, something wasn't. What was working was based on "ps_4_0" not "ps_5_0". And this is probably the main reason from which I need to start implementing and then possibly look for the reason why it doesn't work. By default, the code runs on my Intel 3000 dedicated GPU. And as you see on this topic in this link, these GPUs have different level of features.
<br /><br />
BUT...
<br /><br />
I am uploaded the code from vs5 (https://github.com/KarolDuracz/scratchpad/blob/main/Win32/web_renderer/d3d11_cube_rotation_v5.cpp) which does not work and vs4 (https://github.com/KarolDuracz/scratchpad/blob/main/Win32/web_renderer/d3d11_cube_rotation_v4.cpp) which does work and this image at the bottom is generated from this implementation based on the vs_4_0 shader. I'm using MSVC 2019 now. Even the debugger doesn't handle these types of events. You don't know what's going on. I'm just on my own intuition even if I make a stupid mistake and give CreateWindow instead of CreateWindowsEx which is the first step to not running the entire code and showing the window. 
<br /><br />
This shader code may come from external file but here is implementation inside code as const str* . If you use an external file then the code fragment that loads this shader using CompileShaderFromMemory does not work. You have to do it using D3DCompileFromFile. Instead of the code that compiles this shader from strings, you have to put something like this in this place of the code. This is for vs_5_0. But this is just an example - how to do.
<br /><br />

```
ID3DBlob* VS = nullptr;
    ID3DBlob* PS = nullptr;
    ID3DBlob* errorBlob = nullptr;

    HRESULT hr = D3DCompileFromFile(L"C:\\Users\\kdhome\\source\\repos\\web_renderer\\Debug\\shader.hlsl", nullptr, nullptr, "VShader", "vs_5_0", 0, 0, &VS, &errorBlob);
    if (FAILED(hr)) {
        if (errorBlob) {
            OutputDebugStringA((char*)errorBlob->GetBufferPointer());
            errorBlob->Release();
        }
        return;
    }

    hr = D3DCompileFromFile(L"C:\\Users\\kdhome\\source\\repos\\web_renderer\\Debug\\shader.hlsl", nullptr, nullptr, "PShader", "ps_5_0", 0, 0, &PS, &errorBlob);
    if (FAILED(hr)) {
        if (errorBlob) {
            OutputDebugStringA((char*)errorBlob->GetBufferPointer());
            errorBlob->Release();
        }
        return;
    }

    // Create shaders
    dev->CreateVertexShader(VS->GetBufferPointer(), VS->GetBufferSize(), nullptr, &vertexShader);
    dev->CreatePixelShader(PS->GetBufferPointer(), PS->GetBufferSize(), nullptr, &pixelShader);
```
Shader file souce ```shader.hlsl```

```
cbuffer ConstantBuffer : register(b0)
{
    matrix mvp;
};

struct VS_INPUT
{
    float4 Pos : POSITION;
    float4 Color : COLOR;
};

struct PS_INPUT
{
    float4 Pos : SV_POSITION;
    float4 Color : COLOR;
};

PS_INPUT VShader(VS_INPUT input)
{
    PS_INPUT output = (PS_INPUT)0;
    output.Pos = mul(input.Pos, mvp); // Apply MVP transformation
    output.Color = input.Color;
    return output;
}

float4 PShader(PS_INPUT input) : SV_TARGET
{
    return input.Color; // Output the color
}
```

some old stuff: <br />
https://users.polytech.unice.fr/~buffa/cours/synthese_image/DOCS/trant.sgi.com/opengl/examples/win32_tutorial/win32_tutorial.html <br />
https://learn.microsoft.com/en-us/samples/microsoft/directx-graphics-samples/d3d12-hello-world-samples-win32/ <br />
http://www.directxtutorial.com/Lesson.aspx?lessonid=11-4-5 <br />
https://antongerdelan.net/opengl/d3d11.html <br />

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/Win32/web_renderer/rotating%20cube%20d3d11%20vs4%20not%20vs5%20shader.png?raw=true)

OK, this is not the case about html parser, but intentionally I trapped into this issue. And wrote here some stuff for myself. The main file still doesn't have this weird rendering problem solved. But that's not for now.
<hr>
The goal here is to capture transmission from youtube service to EDGE browser, decode and render on my implementation of HTML parser. But this hard to solve for me right know. And simplest way to do this is to create random generator videos and run on localhost as python server and send data to this my browser engine. And then render on the same time 8 x 8 tiles as thumbnails 100 x 100 px. Which is 16 videos on the page at the same time. And when user click on this tile then user go to video transmission on 600x400 px video size. https://github.com/KarolDuracz/scratchpad/blob/main/Win32/web_renderer/todo_video/README.md
