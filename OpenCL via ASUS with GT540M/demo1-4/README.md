TODO <br /><br />
Some stuff to get access to GPU. Inside this code there is some tests using PTX raw code via OpenCL. And in other hand, using common functions. This is commented inside .cpp files. <br /><br />

Look at line 1092 in scratchpad/OpenCL via ASUS with GT540M/demo1-4/Project12_opencl/Project12_opencl
/main.cpp - // REVERSED VERSION -> load PTX kernel.bin, compile it and run test - PASS :) - this is test to write kernel into .bin file and in next commented code load .bin file and execute. <br /><br />

In project12_opencl from line 214 to 888 PTX code tests. Look at "kernelSource" variables. This needs to be done again here. But looking into file (https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo1-4/Project12_opencl/Project12_opencl/windowsTempkernel.bin) - when you open Project12_opencl .cpp file and when you find "kernel.bin" inside code this is probably some kernel compiled to PTX. But if I remember last time when I do this test, kernel.bin is biggest than this one. And in this .cpp open cl demo there is implementation to write kernel into PTX, and load from .bin (PTX format) to device.<br /><br />

I have "kernel.bin" (https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo1-4/kernel.bin) but this must be compile again using demo "Project12_opencl" and some implementation from inside, to rewrite the kernel Source as PTX (kernel.bin) again. And tests to load kernel.bin file and execute. I put kernel.bin file here, to see what it looks like.<br /><br />

In line 895 there is "// testCoresKernel example" - this is probably test to detect all 96 cores. To fix first implementation (https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/main.cpp#L204) - But what I found on this topic, there is no way (???) to detect all cores like that using opencl. But there are methods like this simple kernel, which detects based on --> int idx = get_global_id(0); - this shows number of threads (???). Or --> (https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo1-4/Project12_opencl/Project12_opencl/main.cpp#L110) to get work group size using clGetKernelWorkGroupInfo() (https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetKernelWorkGroupInfo.html) . Or (https://downloads.ti.com/mctools/esd/docs/opencl/execution/kernels-workgroups-workitems.html)

```
const char* kernelSource = "__kernel void testCoresKernel(__global int* data) {"
"    int idx = get_global_id(0);"
"    data[idx] = idx;"
"}";
```

That's why I used "Project 12 cuda api" and API from  LoadLibrary(L"C:\\CUDA\\bin\\cudart.dll"); to get some information about GPU device.
<br /><br />
But there are more questions here about how the driver itself works... In here I learned how to use API and docs and how to use DLL to do that. Even that requires a better explanation of what I was doing here, but not now. I'm just posting what's there for now. <br />

This kernel probably works and it's doing 99% of the GPU utilization. And it raises the temperature to ~80 C.  (https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo1-4/Project12_opencl/Project12_opencl/main.cpp#L228)

```
// default :
//  int iterations = 10000;
const char* kernelSource =
"__kernel void complex_benchmark("
"    __global float* A, __global float* B, __global float* C, "
"    __global float* D, __global float* E, __global float* F, "
"    const unsigned int n) {"
"    int id = get_global_id(0);"
"    int iterations = 10000;"
"    float accum = 0.0f;"
"    for (int i = 0; i < iterations; i++) {"
"        if (id < n) {"
"            float a = A[id];"
"            float b = B[id];"
"            float c = C[id];"
"            float d = D[id];"
"            float e = E[id];"
"            float f = F[id];"
"            float result1 = (a * b + sin(a + i)) / (cos(b) + 1.0f + i * 0.0001f);"
"            float result2 = sqrt(fabs(a - b + i * 0.001f)) * exp(-a * b * i);"
"            accum += (result1 + result2);"
"            C[id] = result1 + result2 + c; "
"            D[id] = (result1 * result2) / (result1 + result2 + 0.001f + i * 0.0001f);"
"            E[id] = pow(result1, 3) + pow(result2, 2) + e;"
"            F[id] = (C[id] + D[id] + E[id]) / 3.0f + f;"
"            if (i % 100 == 0) {"
"                F[id] += accum; "
"                accum = 0.0f; "
"            }"
"        }"
"    }"
"    if (id < n) {"
"        F[id] += accum; "
"    }"
"}";
```

<h2>How write to kernel.bin as PTX and read - line 1616 and 1204</h2>
In (https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo1-4/Project12_opencl/Project12_opencl/main.cpp#L1204) there is only 1 "fread" function (https://learn.microsoft.com/pl-pl/cpp/c-runtime-library/reference/fread?view=msvc-170) . This means, that is implementation to LOAD kernel.bin. And in other hand, only one "fwrite" in line 1611 (https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo1-4/Project12_opencl/Project12_opencl/main.cpp#L1616) that means, this is implementation how to write to kernel.bin. 

<h2>How to find solution to fix error from second picture "Project 12 cuda api"?</h2>
This is only idea. But when use ReadProcessMemory to manually check stack and each fields of this structure like here (https://github.com/KarolDuracz/scratchpad/blob/main/NtQueryInformationProcess_run_example) --- maybe you find proper structure for this implementation. As you see, I was try to different types but no one works. Current example is work in some way(https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo1-4/Project12_cuda/Project12_cuda/main.cpp#L115) --- So, the solution is to read manually stack process and pass start address to this structure to examine each fields with this diffrent types (size) --- ReadProcessMemory API (https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-readprocessmemory)
<hr>

Project 11
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo1-4/project11.png?raw=true)

Project 12 cuda api
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo1-4/project12cuda.png?raw=true)

Project 12 opencl
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo1-4/project12opencl.png?raw=true)

Project 12 CPU vs GPU - 4 threads test for some computing 
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo1-4/project12cpuvsgpubench.png?raw=true)

<hr> Add more iterations to this kernel to trace utilization

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo1-4/add%20more%20iters.png?raw=true)

<hr>
<br />
[1] https://developer.download.nvidia.com/assets/cuda/files/CUDADownloads/NVML/nvml.pdf <-- Here is guide how to Initialize and Cleanup using nvmlInit() and nvmlShutdown() etc. <br />
[2] https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html <--  random choice topic from docs NVIDIA. _nvmlDeviceQueries. I Used some functions from this list like <br />
https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1g2efc4dd4096173f01d80b2a8bbfd97ad <br />
[3] https://github.com/victusfate/opencl-book-examples/blob/master/src/Chapter_6/HelloBinaryWorld/HelloBinaryWorld.cpp <-- nice example  how to use opencl to retrieving and loading program binaries<br />
