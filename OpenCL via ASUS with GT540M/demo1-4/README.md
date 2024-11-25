TODO <br /><br />
Some stuff to get access to GPU. Inside this code there is some tests using PTX raw code via OpenCL. And in other hand, using common functions. To jest skomentowane inside this .cpp files. <br /><br />

Look at line 1092 in scratchpad/OpenCL via ASUS with GT540M/demo1-4/Project12_opencl/Project12_opencl
/main.cpp - // REVERSED VERSION -> load PTX kernel.bin, compile it and run test - PASS :) - this test write kernel to .bin file and in next commented code load this .bin file and execute. <br /><br />

In this project12_opencl from line 214 to 888 PTX code tests. Look at "kernelSource" variables. This needs to be do again here, but looking into this file (https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo1-4/Project12_opencl/Project12_opencl/windowsTempkernel.bin) - when you open Project12_opencl .cpp file and find "kernel.bin" inside code this is probably some kernel compiled to PTX. But if I remember last time when I do this test, kernel.bin is biggest than this one. And in this .cpp open cl demo there is implementation to write kernel into PTX, and load from .bin (PTX format) to device. But right now I upload only this all stuff here.<br /><br />

In line 895 there is "// testCoresKernel example" - this is probably test to detect al 96 cores. To fix this first implementation (https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/main.cpp#L204) - But what I found on this topic, there is no way (???) to detect all cores like that using opencl. But there are methods like this simple PTX kernel, which detects based on --> int idx = get_global_id(0); number of threads.

```
const char* kernelSource = "__kernel void testCoresKernel(__global int* data) {"
"    int idx = get_global_id(0);"
"    data[idx] = idx;"
"}";
```

That's why I used "Project 12 cuda api" and API from  LoadLibrary(L"C:\\CUDA\\bin\\cudart.dll"); to get some information about GPU device.
<br /><br />
But there are more questions here about how the driver itself works... In here I learned how to use API and docs and how to use DLL to do that. Even that requires a better explanation of what I was doing here, but not now. I'm just posting what's there for now.
<hr>

Project 11
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo1-4/project11.png?raw=true)

Project 12 cuda api
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo1-4/project12cuda.png?raw=true)

Project 12 opencl
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo1-4/project12opencl.png?raw=true)

Project 12 CPU vs GPU - 4 threads test for some computing 
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo1-4/project12cpuvsgpubench.png?raw=true)
