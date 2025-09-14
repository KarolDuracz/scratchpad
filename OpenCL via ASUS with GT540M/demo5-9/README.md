> [!IMPORTANT]  
> These are only fixes for demo1-4. Specifically for the cuda demo "Project 12 cuda api". Nothing else.

> [!IMPORTANT]  
> All demos 5-9 are in one main.cpp file. Below I've listed the line where each demo starts. And it's tagged between #if 0 <> endif keywords. So to run it, you simply need to enter 1 instead of 0 for the listing and recompile it. Currently, the compiled code is at the very end, for demo 5.


<h2>What's going on here - demo 5 - 9</h2>
This code mainly concerns the previous demo from this file. <br /><br />

https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo1-4/Project12_cuda/Project12_cuda/main.cpp

The image titled "Project 12 cuda api" in this folder

https://github.com/KarolDuracz/scratchpad/tree/main/OpenCL%20via%20ASUS%20with%20GT540M/demo1-4

demonstrates how to extract information using CUDA from a GT540M (GF108) device.

This time, using GPT-5, I refactored the code and fixed the structures I had trouble with last time. It seems to be working fine.

<h2>Demo Explanation</h2>

<h3> 1.</h3>  When I first tried to perform some calculations, I had a vague understanding of CUDA in general. Therefore, I had trouble properly detecting the device and retrieving information from it. I also had trouble installing CUDA Toolkit 2.1 (January 2009). I found a workaround by defining types from headers directly in the code using typedef and retrieving function definitions from the official NVIDIA documentation. As can be seen in demos 1-4, I managed to extract some information from CUDA. However, I was also using OpenCL there.
<h3> 2.</h3>  Looking at GPU Cpas Viewer, you can see that it detects multicores x2. However, my code doesn't reflect this. It turned out that I would use the API from nvcuda.dll to extract information, rather than cudart.dll, as I recently tried. (see below for GPT-5 explanation)
<h3> 3.</h3>  So the problem I wanted to solve was recognizing these SM amounts. Access these structures and information correctly. And I managed to do that in this demo using functions from cudart.dll and nvcuda.dll.

I'm not sure if I included the information about where I got the cudart.dll DLL, but I placed the file in the demo1-4 folder. In any case, it's probably from CUDA Toolkit 2.1. I have it installed in:

```
C:\CUDA\bin\cudart.dll
```

Regarding nvcuda.dll, its location is:

```
C:\Windows\System32\nvcuda.dll
```

And it's probably part of the graphics card driver. (I'm not sure now). It's important to note that this is another DLL used here and one I didn't use in previous demos. That's why it's also uploaded here to this repo as a file, weighing ~10 MB.

<h3>4.</h3> My goal was to improve this code and extract the correct information about the number of SMs (which should be 2), and this is what happens in the code on lines 1598-1929. This is also shown in the "demo5" image. The first line uses information from the DLL.

```
ndriver = LoadLibraryW(L"C:\\Windows\\System32\\nvcuda.dll");
```

Only initialization must be performed first, as seen from line 1727.

```
auto myCuInit = (cuInit_t)GetProcAddress(ndriver, "cuInit");
if (!myCuInit) {
std::cerr << "Failed to load cuInit\n";
return 1;
}
```

Without this piece of code that initializes the driver, I immediately try to execute CUresult rc = cudaDeviceGetAttributeFunc(&smCount, attr, dev); from line 1751, I get error 3, which is an initialization error. Therefore, it needs to be run through cuInit first.

Line 1753 shows the result from the first line in this image for demo5.

https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo5-9/Project25%20-%20back%20to%20cuda%20-%2013-09-2025%20-/Project25%20-%20back%20to%20cuda%20-%2013-09-2025%20-/main.cpp#L1598

Because looking at the GF108 device description

https://www.techpowerup.com/gpu-specs/nvidia-gf108.g82

and seeing the block diagram, you can see that it has 2 x SM

<h3>5.</h3> In demos 6-9, I focused on performing memory operations, including scenarios for exchanging data between host system <> GPU device or GPU <> GPU memory. As well as speed measurements, there are also several different scenarios for 99% memory usage, or in a 30-second test, some write sequences to see memory usage and utilization in GPU Caps Viewer.

<h3>Demo pictures</h3>

demo5 - The first version which used GPT-5 involved code refactoring and fixes, and adding a part that loads functions from nvcuda.dll to extract the correct value for the amount of SM

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo5-9/demo5.png?raw=true)

demo6 - https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo5-9/Project25%20-%20back%20to%20cuda%20-%2013-09-2025%20-/Project25%20-%20back%20to%20cuda%20-%2013-09-2025%20-/main.cpp#L1120

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo5-9/demo6.png?raw=true)

demo7 - https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo5-9/Project25%20-%20back%20to%20cuda%20-%2013-09-2025%20-/Project25%20-%20back%20to%20cuda%20-%2013-09-2025%20-/main.cpp#L754

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo5-9/demo7.png?raw=true)

demo8 - https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo5-9/Project25%20-%20back%20to%20cuda%20-%2013-09-2025%20-/Project25%20-%20back%20to%20cuda%20-%2013-09-2025%20-/main.cpp#L345

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo5-9/demo8.png?raw=true)

demo 9 - not working - https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo5-9/Project25%20-%20back%20to%20cuda%20-%2013-09-2025%20-/Project25%20-%20back%20to%20cuda%20-%2013-09-2025%20-/main.cpp#L3

<br /><br />
A little explanation from GPT-5 for point 2

<hr>

```
You're mixing **CUDA Runtime API** (`cudaGetDeviceProperties`) with the **CUDA Driver API** (`cuDeviceGetAttribute`).
They're different APIs, different DLLs:

* **Runtime API** lives in `cudart.dll`
* **Driver API** lives in `nvcuda.dll`

The function you want (`cuDeviceGetAttribute`) is part of the **Driver API**.
So your typedef must match the Driver API signature, not `cudaError_t`.

---

 Official signature (from `cuda.h`)

```c
CUresult cuDeviceGetAttribute(int *pi, 
CUdevice_attribute attrib, 
CUdevicedev);
```

Definitions:

* `CUresult` is an `enum` (error codes) → you can typedef it as `int`.
* `CUdevice` is also an `int` (device ordinal).
* `CUdevice_attribute` is an enum (lots of attributes, e.g. `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`).

 Typedef for dynamic loading

```cpp
typedef int CResult; // driver API result
typedef int CUdevice; // device handle (ordinal)
typedef int CUdevice_attribute; // attribute ID

typedef CResult(__stdcall* cuDeviceGetAttribute_t)(int* pi, 
CUdevice_attribute attrib, 
CUdevicedev);
```

 Example usage

```cpp
// Load the driver API
HMODULE h = LoadLibraryW(L"nvcuda.dll");
if (!h) { 
std::cerr << "Failed to load nvcuda.dll\n"; 
return 1;
}

auto myCuDeviceGetAttribute = (cuDeviceGetAttribute_t)GetProcAddress(h, "cuDeviceGetAttribute");
if (!myCuDeviceGetAttribute) { 
std::cerr << "Failed to load cuDeviceGetAttribute\n"; 
return 1;
}

int smCount = 0;
CUdevicedev = 0; // firstGPU
CUdevice_attribute attr = 16; // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT (value = 16 in headers)

CUresult rc = myCuDeviceGetAttribute(&smCount, attr, dev);
if (rc == 0) { 
std::cout << "Multiprocessor count = " << smCount << "\n";
} else { 
std::cerr << "cuDeviceGetAttribute failed with code " << rc << "\n";
}
```

👉 Question for you: do you want me to also give you the **enum values** for `CUdevice_attribute` (like `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16`) so you don't need CUDA headers, or will you copy them from `cuda.h`?



<hr>

<h3>References</h3>
1. https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDA__DEVICE_g9c3e1414f0ad901d3278a4d6645fc266.html#g9c3e1414f0ad901d3278a4d6645fc266 <br />
2. https://www.techpowerup.com/gpu-specs/nvidia-gf108.g82#gallery-5 <br />
3. https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html << official documentation with parameters e.g. CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT

