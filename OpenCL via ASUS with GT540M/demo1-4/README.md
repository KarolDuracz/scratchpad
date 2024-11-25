TODO <br /><br />
Some stuff to get access to GPU. Inside this code there is some tests using PTX raw code via OpenCL. And in other hand, using common functions. To jest skomentowane inside this .cpp files. <br /><br />

Look at line 1092 in scratchpad/OpenCL via ASUS with GT540M/demo1-4/Project12_opencl/Project12_opencl
/main.cpp - // REVERSED VERSION -> load PTX kernel.bin, compile it and run test - PASS :) - this test write kernel to .bin file and in next commented code load this .bin file and execute. <br /><br />

In this project12_opencl from line 214 to 888 PTX code tests. Look at "kernelSource" variables <br /><br />

In line 895 there is "// testCoresKernel example" - this is probably test to detect al 96 cores. To fix this first implementation (https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/main.cpp#L204)
<hr>

Project 11
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo1-4/project11.png?raw=true)

Project 12 cuda api
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo1-4/project12cuda.png?raw=true)

Project 12 opencl
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo1-4/project12opencl.png?raw=true)

Project 12 CPU vs GPU - 4 threads test for some computing 
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/OpenCL%20via%20ASUS%20with%20GT540M/demo1-4/project12cpuvsgpubench.png?raw=true)
