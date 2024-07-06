//#include <Windows.h>
//#include <stdio.h>
//#include <stdlib.h>
//
//int main()
//{
//
//	const char str1[] = "this is thes 123 0 sakd ask emain@karol aaa hehe";
//	const char str2[] = "1";
//	const char *p = strstr(str1, str2);
//	printf("%s %p \n", p, p);
//
//	const char** _argv = (const char**)malloc(2);
//
//	_argv[0] = "hello\n";
//	_argv[1] =  "test" ;
//
//	printf("%s \n", _argv[0]);
//
//	return 0;
//}

#include <stdio.h>
#include <stdint.h>


typedef union {
    float f;
    struct {
        uint32_t mantissa : 23;
        uint32_t exponent : 8;
        uint32_t sign : 1;
    } parts;
} ieee_float;

// Function to print binary representation of a float
void print_binary(ieee_float num) {
    printf("Sign: %u, Exponent: %u, Mantissa: %u\n", num.parts.sign, num.parts.exponent, num.parts.mantissa);
}

// Utility function to add leading 1 in the mantissa
uint32_t get_full_mantissa(ieee_float num) {
    return (1 << 23) | num.parts.mantissa;
}

// Align exponents by shifting the mantissa of the number with the smaller exponent
void align_exponents(ieee_float* num1, ieee_float* num2) {
    if (num1->parts.exponent > num2->parts.exponent) {
        int shift = num1->parts.exponent - num2->parts.exponent;
        num2->parts.mantissa >>= shift;
        num2->parts.exponent += shift;
    }
    else if (num1->parts.exponent < num2->parts.exponent) {
        int shift = num2->parts.exponent - num1->parts.exponent;
        num1->parts.mantissa >>= shift;
        num1->parts.exponent += shift;
    }
}

// Function to add two floating-point numbers
ieee_float add_floats(ieee_float num1, ieee_float num2) {
    // Align exponents
    align_exponents(&num1, &num2);

    // Add mantissas
    uint32_t mantissa1 = get_full_mantissa(num1);
    uint32_t mantissa2 = get_full_mantissa(num2);
    uint32_t result_mantissa = mantissa1 + mantissa2;

    // Normalize the result
    int result_exponent = num1.parts.exponent;
    if (result_mantissa & (1 << 24)) {
        result_mantissa >>= 1;
        result_exponent++;
    }

    // Create result
    ieee_float result;
    result.parts.sign = num1.parts.sign;  // Assuming both have the same sign for simplicity
    result.parts.exponent = result_exponent;
    result.parts.mantissa = result_mantissa & ((1 << 23) - 1);

    printf("%g %f %x\n", result, (float)result.f, result);

    return result;
}

#include <Windows.h>
#include <CL/cl.h>

// Simple compute kernel which computes the square of an input array 
//
const char* KernelSource = "\n" \
"__kernel void square(                                                       \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = input[i] * input[i];                                \n" \
"}                                                                      \n" \
"\n";

//const char* KernelSource = "#if defined(cl_khr_fp64)\n"\
//"#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"\
//"#elif defined(cl_amd_fp64)\n"\
//"#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"\
//"#else\n"\
//"#  error double precision is not supported\n"\
//"#endif\n"\
//"kernel void add(\n"\
//"       ulong n,\n"\
//"       global const double *a,\n"\
//"       global const double *b,\n"\
//"       global double *c\n"\
//"       )\n"\
//"{\n"\
//"    size_t i = get_global_id(0);\n"\
//"    if (i < n) {\n"\
//"       c[i] = a[i] + b[i];\n"\
//"    }\n"\
//"}\n";

typedef HRESULT(CALLBACK* LPFNDLLFUNC1)(cl_uint, cl_platform_id*, cl_uint*); // clGetPlatformIDs
typedef HRESULT(CALLBACK* LPFNDLLFUNC2)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*); // clGetDeviceIDs
typedef cl_context(CALLBACK* LP_clCreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*, void (CL_CALLBACK*)(const char* , const void* , size_t , void*), void*, cl_int*); // clCreateContext
typedef cl_int(CALLBACK* LP_clGetDeviceInfo)(cl_device_id, cl_device_info, size_t, void*, size_t*); 
typedef cl_command_queue(CALLBACK* LP_clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
typedef cl_program(CALLBACK* LP_clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int(CALLBACK* LP_clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void (CL_CALLBACK* )(cl_program , void* ), void*);
typedef cl_kernel(CALLBACK* LP_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_mem(CALLBACK* LP_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
typedef cl_int(CALLBACK* LP_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int(CALLBACK* LP_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int(CALLBACK* LP_clGetKernelWorkGroupInfo)(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void*, size_t);
typedef cl_int(CALLBACK* LP_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int(CALLBACK* LP_clFinish)(cl_command_queue);
typedef cl_int(CALLBACK* LP_clEnqueueReadBuffer)(cl_command_queue , cl_mem , cl_bool ,   size_t , size_t ,  void* , cl_uint ,const cl_event* ,cl_event* );

// https://github.com/rsnemmen/OpenCL-examples/blob/master/add_numbers/add_numbers.c

volatile int uniq_ids = 0;

int main() {

    HINSTANCE dll;
    dll = LoadLibrary(L"C:\\Program Files\\NVIDIA Corporation\\OpenCL\\OpenCL.dll");
    LPFNDLLFUNC1 lpfnDllFunc1;    // Function pointer
    cl_uint num = 0;
    cl_int CL_err = CL_SUCCESS;

    cl_device_id buf[100]; // test allocate 96 cores

    if (dll != NULL) {
        lpfnDllFunc1 = (LPFNDLLFUNC1)GetProcAddress(dll , "clGetPlatformIDs");
        LPFNDLLFUNC2 lp2 = (LPFNDLLFUNC2)GetProcAddress(dll, "clGetDeviceIDs");
        LP_clCreateContext lp_ctx = (LP_clCreateContext)GetProcAddress(dll, "clCreateContext");
        LP_clGetDeviceInfo lc_info = (LP_clGetDeviceInfo)GetProcAddress(dll, "clGetDeviceInfo");
        LP_clCreateCommandQueue lc_cmd_queue = (LP_clCreateCommandQueue)GetProcAddress(dll, "clCreateCommandQueue");
        LP_clCreateProgramWithSource lc_cpws = (LP_clCreateProgramWithSource)GetProcAddress(dll, "clCreateProgramWithSource");
        LP_clBuildProgram lc_cbuild = (LP_clBuildProgram)GetProcAddress(dll, "clBuildProgram");
        LP_clCreateKernel lc_kernel = (LP_clCreateKernel)GetProcAddress(dll, "clCreateKernel");
        LP_clCreateBuffer lc_cbuf = (LP_clCreateBuffer)GetProcAddress(dll, "clCreateBuffer");
        LP_clEnqueueWriteBuffer lc_eb = (LP_clEnqueueWriteBuffer)GetProcAddress(dll, "clEnqueueWriteBuffer");
        LP_clSetKernelArg lc_ska = (LP_clSetKernelArg)GetProcAddress(dll, "clSetKernelArg");
        LP_clGetKernelWorkGroupInfo lc_gkwgi = (LP_clGetKernelWorkGroupInfo)GetProcAddress(dll, "clGetKernelWorkGroupInfo");
        LP_clEnqueueNDRangeKernel lc_erk = (LP_clEnqueueNDRangeKernel)GetProcAddress(dll, "clEnqueueNDRangeKernel");
        LP_clFinish lc_finish = (LP_clFinish)GetProcAddress(dll, "clFinish");
        LP_clEnqueueReadBuffer lc_erb = (LP_clEnqueueReadBuffer)GetProcAddress(dll, "clEnqueueReadBuffer");

        // second test - get platform id
        cl_platform_id platform;
        cl_device_id dev;

        //printf("%d \n", dll);
        if (lpfnDllFunc1 != NULL) {
            CL_err = lpfnDllFunc1(0, NULL, &num);
            CL_err = lpfnDllFunc1(1, &platform, NULL);
            printf("%d %d %d %d \n", CL_err, num, CL_SUCCESS, platform);
            CL_err = lp2(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
            printf("%x \n ", dev);
            if (CL_err == CL_DEVICE_NOT_FOUND) {
                printf("can only run on cpu \n");
            }
            // get device info
            cl_device_type type;
            cl_uint vendor;
            size_t ret = 0;
            size_t ret_vendor = 0;
            CL_err = lc_info(dev, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, &ret);
            CL_err = lc_info(dev, CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &vendor, &ret_vendor);
            printf("%d %d %d %d %d \n", CL_err, ret, type, vendor, ret_vendor);

            // create context 
            cl_context context;
            context = lp_ctx(NULL, 1, &dev, NULL, NULL, &CL_err);
            if (CL_err < 0) {
                perror("couldn't craete context");
                exit(1);
            }

            while (1) {
                for (int i = 0; i < 100; i++) {
                    if (buf[i] != dev) {
                        /*buf[i] = dev;
                        printf("%d \n", dev);
                        uniq_ids++;*/
                        goto __alloc_to_buf;
                    }
                }
            __alloc_to_buf:
                buf[uniq_ids] = dev;
                uniq_ids++;
                if (uniq_ids >= 96) {
                    break;
                }
            }

            printf(" unique ids %d \n", uniq_ids);

            // from this all copied from https://github.com/jcupitt/opencl-experiments/blob/master/OpenCL_Hello_World_Example/hello.c
            // craete a command commans
            cl_command_queue commands;
            commands = lc_cmd_queue(context, dev, 0, &CL_err);
            if (!commands) {
                printf("error 1\n");
                return 1; // exit(1);
            }

            //. create the computer program form the souce buffer
            // !!!
            cl_program program;
            program = lc_cpws(context, 1, (const char**)&KernelSource, NULL, &CL_err);
            if (!program) {
                printf("error 2\n");
                return 1; // exit(1);
            }

            // build the program executable
            // https://github.com/jcupitt/opencl-experiments/blob/master/OpenCL_Hello_World_Example/hello.c
            CL_err = lc_cbuild(program, 0, NULL, NULL, NULL, NULL);
            if (CL_err != CL_SUCCESS) {
                printf("errrrrror 3\n");
                return 1;

                /*
                 err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
                if (err != CL_SUCCESS)
                {
                    size_t len;
                    char buffer[2048];

                    printf("Error: Failed to build program executable!\n");
                    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
                    printf("%s\n", buffer);
                    exit(1);
                }
                */

            }

            // craete kernel 
            cl_kernel kernel;
            kernel = lc_kernel(program, "square", &CL_err);
            if (!kernel || CL_err != CL_SUCCESS) {
                printf("error 4");
                return 1;
            }

            // Create the input and output arrays in device memory for our calculation
            #define DATA_SIZE (1024)
            float data[DATA_SIZE];              // original data set given to device
            int i = 0;
            unsigned int count = DATA_SIZE;
            for (i = 0; i < count; i++)
                data[i] = rand() / (float)RAND_MAX;
            ////////////////////////////////////////////////////////////////////////////////////////////////
            

            cl_mem input;                       // device memory used for the input array
            cl_mem output;                      // device memory used for the output array
            input = lc_cbuf(context, CL_MEM_READ_ONLY, sizeof(float) * count, NULL, NULL);
            output = lc_cbuf(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
            if (!input || !output)
            {
                printf("Error: Failed to allocate device memory!\n");
                return 1;
            }

            // write our data set into the input aray in device mmeory
            CL_err - lc_eb(commands, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
            if (CL_err != CL_SUCCESS)
            {
                printf("Error: Failed\n");
                return 1;
            }

            // Set the arguments to our compute kernel
            CL_err = 0;
            CL_err = lc_ska(kernel, 0, sizeof(cl_mem), &input);
            CL_err |= lc_ska(kernel, 1, sizeof(cl_mem), &output);
            CL_err |= lc_ska(kernel, 2, sizeof(unsigned int), &count);
            if (CL_err != CL_SUCCESS)
            {
                printf("Error: Failed to set kernel arguments! %d\n", CL_err);
                exit(1);
            }

            //
            //lc_gkwgi
            size_t local;
            CL_err = lc_gkwgi(kernel, dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
            if (CL_err != CL_SUCCESS)
            {
                printf("Error: Failed to retrieve kernel work group info! %d\n", CL_err);
                exit(1);
            }

            // 
            size_t global;
            global = count;
            //lc_erk
            global = count;
            CL_err = lc_erk(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
            if (CL_err)
            {
                printf("Error: Failed to execute kernel!\n");
                return EXIT_FAILURE;
            }

            // finish
            lc_finish(commands);

            // Read back the results from the device to verify the output
            float results[DATA_SIZE];
            CL_err = lc_erb(commands, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL);
            if (CL_err != CL_SUCCESS)
            {
                printf("Error: Failed to read output array! %d\n", CL_err);
                exit(1);
            }

            // valiudate our res
            printf("---> test \n");
            unsigned int correct = 0;
            for (i = 0; i < count; i++)
            {
                if (results[i] == data[i] * data[i])
                    correct++;
            }

            printf("Computed '%d/%d' correct values!\n", correct, count);

            /*
            * https://github.com/jcupitt/opencl-experiments/blob/master/OpenCL_Hello_World_Example/hello.c
            * https://gist.github.com/ddemidov/2925717
            * https://github.com/KhronosGroup/OpenCL-Guide/blob/main/chapters/getting_started_windows.md
            * https://github.com/rsnemmen/OpenCL-examples/blob/master/add_numbers/add_numbers.c
            * https://github.com/intel/clGPU/tree/master/common/intel_ocl_icd/windows/Debug/lib/x64
            * https://www.intel.com/content/www/us/en/developer/articles/tool/opencl-drivers.html
            * https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#CL_DEVICE_TYPE_GPU
            * https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetDeviceIDs.html
            * https://stackoverflow.com/questions/10852696/opencl-device-uniqueness
            * https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html
            * https://community.intel.com/t5/OpenCL-for-CPU/OpenCL-CPU-runtime-cannot-find-any-platforms/td-p/1408317
            clReleaseMemObject(input);
            clReleaseMemObject(output);
            clReleaseProgram(program);
            clReleaseKernel(kernel);
            clReleaseCommandQueue(commands);
            clReleaseContext(context);
            */
        }
        else {
            printf("load atach function \n");
        }
        FreeLibrary(dll);
    }
    else {
        printf("cant loaded library \n");
    }

   /* cl_uint num = 0;
    cl_int CL_err = CL_SUCCESS;
    CL_err = clGetPlatformIDs(0, NULL, &num);
    if (CL_err = CL_SUCCESS) {
        printf("works %d", CL_err);
    }
    else {
        printf("not works %d", CL_err);
    }*/

    return 0;

    /*ieee_float num1, num2, result;

    num1.f = 3.5;
    num2.f = 2.25;

    printf("Number 1:\n");
    print_binary(num1);
    printf("%f \n", num1.f);

    printf("Number 2:\n");
    print_binary(num2);
    printf("%f \n", num2.f);

    result = add_floats(num1, num2);

    printf("Result:\n");
    print_binary(result);
    printf("%f \n", result.f);

    return 0;*/
}
