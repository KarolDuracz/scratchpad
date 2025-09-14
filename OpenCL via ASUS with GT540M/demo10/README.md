> [!WARNING]  
> This is unfinished. The implementation is only intended to create a base code that works, at least to a small extent.



<h2>Demo 10</h2>
This isn't the best choice for a demo that will be running a stress test. But it's an example of mixing CUDA functions with an application that renders some simple animation. This is a continuation of Demo5-9 using the APIs from nvcuda.dll and cudart.dll. And it's not 3D, as the name suggests. But it does provide some input into the next steps, like adding mouse, keyboard, and 3D support in the future.
<br /><br />
Demo 1 - https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo10/Project25%20-%20back%20to%20cuda%20-%203d%20application/Project25%20-%20back%20to%20cuda%20-%203d%20application/main.cpp#L2 <br /><br />

Demo 2 - https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo10/Project25%20-%20back%20to%20cuda%20-%203d%20application/Project25%20-%20back%20to%20cuda%20-%203d%20application/main.cpp#L575<br /><br />

Demo 3 - https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo10/Project25%20-%20back%20to%20cuda%20-%203d%20application/Project25%20-%20back%20to%20cuda%20-%203d%20application/main.cpp#L1110<br /><br />

This is an example from the code at the very top in main.cpp. It uses very little GPU and memory. <b>In the upper left corner is the "Objects" menu. </b> Add individual items there, or enable automatic addition of 100 every second.

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo10/demo10.png?raw=true)

Demo 3 - Here's an attempt to implement Host <> GPU data transfer rate measurements. Needs some testing. TODO.

![dumpp](https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo10/demo10%20-%20example%203.png?raw=true)

<h2>3D demo + CUDA acceleration</h2>

What's improved? Added 3D objects, more object types like 3D boxes and 3D circles. Ability to zoom in/out with the mouse and rotate the camera to some extent.

demo 3 - https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo10/Project25%20-%20back%20to%20cuda%20-%203d%20application%20-%20demo2%2014092025/Project25%20-%20back%20to%20cuda%20-%203d%20application%20-%20demo2%2014092025/main.cpp#L1

demo 2 - https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo10/Project25%20-%20back%20to%20cuda%20-%203d%20application%20-%20demo2%2014092025/Project25%20-%20back%20to%20cuda%20-%203d%20application%20-%20demo2%2014092025/main.cpp#L654

demo 1 - https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo10/Project25%20-%20back%20to%20cuda%20-%203d%20application%20-%20demo2%2014092025/Project25%20-%20back%20to%20cuda%20-%203d%20application%20-%20demo2%2014092025/main.cpp#L1699

Issue #1 - Demo 3 needs fixes that actually create an implementation that directly uses the CUDA API, which provides true hardware acceleration. Currently, the test fails and is handled by the CPU. This applies to the image (demo3d - demo 3.png). A few ways to go:

```
When `gpu_build_polygons_driver_api` returns `false` it usually happened at one of these steps:

* `nvcuda.dll` not found (NVIDIA driver missing).
* `GetProcAddress` failed for required driver API symbols (driver too old / function name variations).
* `cuInit(0)` failed (driver problem or incompatible driver).
* `cuDeviceGetCount` reported 0 devices (no GPU visible).
* `cuCtxCreate` failed — context couldn't be created.
* Module loading failed (`cuModuleLoadDataEx`) — usually because `embedded_kernel.ptx` is missing, or PTX/CUBIN is invalid or targeted at wrong compute capability, or the PTX text is corrupted.
* `cuModuleGetFunction` failed — kernel name in module doesn't match (e.g. you asked for `transform_polys` but the PTX exports `transform_polys_kernel`).
* Memory allocations or host<>device copies failed.
* You accidentally compiled a stub version (the safe stub I suggested earlier returns false), or the definition with the real implementation isn't actually linked (duplicate or missing file).
```

Issue #2 - For demo 3, meaning the code from the top down, I might need to change the subsystem to CONSOLE in MSVC (2019 in my case). Maybe also from x86 to x64. I haven't tested this thoroughly, but for demo 3, I think I need to change the subsystem to CONSOLE. For demo 2, it's already subsystem:WINDOWS. So, Project Properties > Linker > System > subsystem field.
<br /><br />

In this image, communication with the CUDA API is implemented to some extent, as you can see in the panel that it tries to display some information about memory transfers. But this is only partially done, trying

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo10/demo3d%20-%20demo%202.png?raw=true)

Attempting to apply CUDA acceleration, but the code isn't finished. See above for possible fixes to get you started. That is, changing key parts of the code to an implementation that directly uses functions from the CUDA API to achieve hardware acceleration. TODO. 

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo10/demo3d%20-%20demo%203.png?raw=true)

