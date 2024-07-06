1. MS Visual Studio setup<br />
Project property >  C/C++ > General > Aditional Include Directories<br />
  {path_to_cl.h}\OpenCL-SDK\clGPU-master\clGPU-master\common\intel_ocl_icd\windows\include<br />
  <br />
2. Don't need .lib put in linker section because in main.cpp directly load all OpenCL functions usin LoadLibrary <br />
3. This dll files comes from nvidia driver C:\Program Files\NVIDIA Corporation\OpenCL . I don't remember exactly when I install this. But this is from here.  <br />
4. File "raw FPU computing" is to check the calculations with raw FPU implementation <br /><br />

![nvidiaGT540M_opencl_access.png](https://raw.githubusercontent.com/KarolDuracz/scratchpad/main/OpenCL%20via%20ASUS%20with%20GT540M/nvidiaGT540M_opencl_access.png)
