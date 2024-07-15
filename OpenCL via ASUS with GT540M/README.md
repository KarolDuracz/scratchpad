Update 15-07-2024.<br />
"don't need .lib" I mean ... i don't have .lib file. So I used this way to run functions from DLL file directly used LoadLibrary.<br />
<br />
1. MS Visual Studio setup<br />
Project property >  C/C++ > General > Aditional Include Directories<br />
  {path_to_cl.h}\OpenCL-SDK\clGPU-master\clGPU-master\common\intel_ocl_icd\windows\include<br />
  <br />
2. Don't need .lib put in linker section because in main.cpp directly load all OpenCL functions using LoadLibrary <br />
3. This dll files comes from nvidia driver C:\Program Files\NVIDIA Corporation\OpenCL . I don't remember exactly when I installed this. But this is from here.  <br />
4. File "raw FPU computing" is to check the calculations with raw FPU implementation <br />
5. This test for detect all cores in main.cpp is wrong heh. Not working but at this moment doesn't matter. Next time check this.<br /><br />

![nvidiaGT540M_opencl_access.png](https://raw.githubusercontent.com/KarolDuracz/scratchpad/main/OpenCL%20via%20ASUS%20with%20GT540M/nvidiaGT540M_opencl_access.png)
