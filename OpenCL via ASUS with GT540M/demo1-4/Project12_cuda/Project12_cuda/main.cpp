#include <windows.h>
#include <iostream>

typedef enum enumcudaError {
    cudaSuccess = 0, 
    cudaErrorInvalidValue = 1
} enumcudaError;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    //Host->Host
    cudaMemcpyHostToDevice = 1,
    //Host->Device
    cudaMemcpyDeviceToHost = 2,
    //Device->Host
    cudaMemcpyDeviceToDevice = 3,
    //Device->Device
    cudaMemcpyDefault = 4
    //Direction of the transfer is inferred from the pointer values.Requires unified virtual addressing
};

typedef enumcudaError cudaError_t;

// Enum for cudaMemcpyKind
//enum cudaMemcpyKind {
//    cudaMemcpyHostToDevice = 1,
//    cudaMemcpyDeviceToHost = 2
//};

/*
struct cudaDeviceProp {
    char name[256];               // Device name
    size_t totalGlobalMem;         // Total global memory
    int multiProcessorCount;       // Number of multiprocessors
    int clockRate;                 // Core clock rate (in kilohertz)
    int memoryClockRate;           // Memory clock rate (in kilohertz)
    int memoryBusWidth;            // Memory bus width (in bits)
    int major;                     // Major compute capability
    int minor;                     // Minor compute capability
    size_t totalConstMem;          // Total constant memory
};
*/

/*
struct cudaDeviceProp {
    char name[256];               // Device name
    size_t totalGlobalMem;         // Total amount of global memory available
    size_t sharedMemPerBlock;      // Amount of shared memory available per block
    int regsPerBlock;              // Number of 32-bit registers available per block
    int warpSize;                  // Warp size in threads
    size_t memPitch;               // Maximum pitch in bytes allowed by memory copies
    int maxThreadsPerBlock;        // Maximum number of threads per block
    int maxThreadsDim[3];          // Maximum size of each dimension of a block
    int maxGridSize[3];            // Maximum size of each dimension of a grid
    int clockRate;                 // Clock frequency in kilohertz
    size_t totalConstMem;          // Total amount of constant memory available on the device
    int major;                     // Major compute capability
    int minor;                     // Minor compute capability
    size_t textureAlignment;       // Alignment requirement for textures
    int multiProcessorCount;       // Number of multiprocessors on the device
    int kernelExecTimeoutEnabled;  // Whether there is a run time limit on kernels
    int memoryClockRate;           // Memory clock frequency in kilohertz
    int memoryBusWidth;            // Memory bus width in bits
    size_t l2CacheSize;            // L2 cache size
    int maxThreadsPerMultiProcessor;// Maximum number of threads per multiprocessor
};
*/

/*
struct cudaDeviceProp {
    char name[256];                   // Name of the device
    size_t totalGlobalMem;             // Total amount of global memory available on the device
    size_t sharedMemPerBlock;          // Amount of shared memory available per block in bytes
    int regsPerBlock;                  // Number of 32-bit registers available per block
    int warpSize;                      // Warp size in threads
    size_t memPitch;                   // Maximum pitch in bytes allowed by memory copies
    int maxThreadsPerBlock;            // Maximum number of threads per block
    int maxThreadsDim[3];              // Maximum size of each dimension of a block
    int maxGridSize[3];                // Maximum size of each dimension of a grid
    int clockRate;                     // Clock frequency in kilohertz
    size_t totalConstMem;              // Total amount of constant memory available on the device
    int major;                         // Major compute capability
    int minor;                         // Minor compute capability
    size_t textureAlignment;           // Alignment requirement for textures
    size_t texturePitchAlignment;      // Pitch alignment requirement for texture references bound to pitched memory
    int multiProcessorCount;           // Number of multiprocessors on the device
    int kernelExecTimeoutEnabled;      // Specified whether there is a run time limit on kernels
    int integrated;                    // Device is integrated with the host
    int canMapHostMemory;              // Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer
    int computeMode;                   // Compute mode (See cudaComputeMode)
    int concurrentKernels;             // Device can possibly execute multiple kernels concurrently
    int ECCEnabled;                    // Device has ECC support enabled
    int pciBusID;                      // PCI bus ID of the device
    int pciDeviceID;                   // PCI device ID of the device
    int pciDomainID;                   // PCI domain ID of the device
    int tccDriver;                     // 1 if device is a Tesla device using TCC driver, 0 otherwise
    int asyncEngineCount;              // Number of asynchronous engines
    int unifiedAddressing;             // Device shares a unified address space with the host
    int memoryClockRate;               // Peak memory clock frequency in kilohertz
    int memoryBusWidth;                // Global memory bus width in bits
    int l2CacheSize;                   // Size of L2 cache in bytes
    int maxThreadsPerMultiProcessor;   // Maximum resident threads per multiprocessor
    int streamPrioritiesSupported;     // Whether device supports stream priorities
    int globalL1CacheSupported;        // Device supports caching globals in L1
    int localL1CacheSupported;         // Device supports caching locals in L1
    size_t sharedMemPerMultiprocessor; // Shared memory available per multiprocessor in bytes
    int regsPerMultiprocessor;         // 32-bit registers available per multiprocessor
    int managedMemory;                 // Device supports allocating managed memory on this system
    int isMultiGpuBoard;               // Device is on a multi-GPU board
    int multiGpuBoardGroupID;          // Unique identifier for a group of devices on the same multi-GPU board
};

*/

struct cudaDeviceProp {
    char name[256];                // Device name
    size_t totalGlobalMem;          // Total global memory available in bytes
    size_t sharedMemPerBlock;       // Amount of shared memory available per block in bytes
    int regsPerBlock;               // Number of 32-bit registers available per block
    int warpSize;                   // Warp size in threads
    size_t memPitch;                // Maximum pitch allowed by memory copies in bytes
    int maxThreadsPerBlock;         // Maximum number of threads per block
    int maxThreadsDim[3];           // Maximum size of each dimension of a block (width, height, depth)
    int maxGridSize[3];             // Maximum size of each dimension of a grid (width, height, depth)
    int clockRate;                  // Clock frequency in kilohertz
    size_t totalConstMem;           // Total constant memory available in bytes
    int major;                      // Major compute capability
    int minor;                      // Minor compute capability
    size_t textureAlignment;        // Alignment requirement for textures
    int multiProcessorCount;        // Number of multiprocessors on the device
    int kernelExecTimeoutEnabled;   // Run time limit on kernels
    int integrated;                 // Integrated with host memory
    int canMapHostMemory;           // Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer
    int computeMode;                // Compute mode (default, exclusive, etc.)
    int concurrentKernels;          // Device supports concurrent kernel execution
    int ECCEnabled;                 // Device has ECC support
    int pciBusID;                   // PCI Bus ID
    int pciDeviceID;                // PCI Device ID
    int tccDriver;                  // Device uses TCC driver (Tesla Compute Cluster)
};

// 
struct gdev_list {
    struct gdev_list* next;
    struct gdev_list* prev;
    void* container;
};


//typedef CUctx_st* CUcontext;

// Function pointer typedefs based on known CUDA function signatures
typedef cudaError_t(*cudaMalloc_t)(void**, size_t);
typedef cudaError_t(*cudaFree_t)(void*);
typedef cudaError_t(*cudaMemcpy_t)(void*, const void*, size_t, cudaMemcpyKind);
typedef cudaError_t(*cudaGetErrorString_t)(cudaError_t, const char**);

typedef cudaError_t(*cudaGetDeviceProperties_t)(cudaDeviceProp* prop, int device);
//typedef cudaError_t(*cuCtxCreate)(CUcontext* pctx, unsigned int  flags, CUdevice dev)

// Function to load cudart.dll and retrieve the function addresses
void main() {
    // Load cudart.dll
    HMODULE hCuda = LoadLibrary(L"C:\\CUDA\\bin\\cudart.dll");
    if (!hCuda) {
        std::cerr << "Failed to load cudart.dll!" << GetLastError() << std::endl;
        return;
    }

    // Retrieve the function pointers using GetProcAddress
    cudaMalloc_t myCudaMalloc = (cudaMalloc_t)GetProcAddress(hCuda, "cudaMalloc");
    cudaFree_t myCudaFree = (cudaFree_t)GetProcAddress(hCuda, "cudaFree");
    cudaMemcpy_t myCudaMemcpy = (cudaMemcpy_t)GetProcAddress(hCuda, "cudaMemcpy");
    cudaGetErrorString_t myCudaGetErrorString = (cudaGetErrorString_t)GetProcAddress(hCuda, "cudaGetErrorString");

    if (!myCudaMalloc || !myCudaFree || !myCudaMemcpy || !myCudaGetErrorString) {
        std::cerr << "Failed to load one or more functions from cudart.dll!" << std::endl;
        FreeLibrary(hCuda); // Clean up
        return;
    }

    // Now you can use the CUDA functions via the function pointers
    // Example: Allocate memory on the GPU
    void* devPtr;
    size_t size = 1024;
    cudaError_t err = myCudaMalloc(&devPtr, size);
    if (err != 0) {  // CUDA_SUCCESS is 0
        std::cerr << "cudaMalloc failed!" << std::endl;
        FreeLibrary(hCuda);
        return;
    }

    std::cout << "cudaMalloc succeeded, allocated 1024 bytes on GPU." << std::endl;

    // Use the allocated memory, for example, copy data to it
    int hostData[256] = { 0 }; // Array on the host
    err = myCudaMemcpy(devPtr, hostData, size, cudaMemcpyHostToDevice);
    if (err != 0) {
        std::cerr << "cudaMemcpy failed!" << std::endl;
    }
    else {
        std::cout << "cudaMemcpy succeeded, copied data to GPU." << std::endl;
    }

    // Free GPU memory
    err = myCudaFree(devPtr);
    if (err != 0) {
        std::cerr << "cudaFree failed!" << std::endl;
    }
    else {
        std::cout << "cudaFree succeeded." << std::endl;
    }

    // Get cudaGetDeviceProperties function
    cudaGetDeviceProperties_t myCudaGetDeviceProperties = (cudaGetDeviceProperties_t)GetProcAddress(hCuda, "cudaGetDeviceProperties");
    if (!myCudaGetDeviceProperties) {
        std::cerr << "Failed to load cudaGetDeviceProperties from cudart.dll!" << std::endl;
        FreeLibrary(hCuda);
        return;
    }

    // Query the number of CUDA devices
    int deviceCount = 0;
    typedef cudaError_t(*cudaGetDeviceCount_t)(int*);
    cudaGetDeviceCount_t myCudaGetDeviceCount = (cudaGetDeviceCount_t)GetProcAddress(hCuda, "cudaGetDeviceCount");
    if (myCudaGetDeviceCount) {
        myCudaGetDeviceCount(&deviceCount);
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        FreeLibrary(hCuda);
        return;
    }

    // cuda detect all cores 
     // Create an array to store results
    //const int numMultiprocessors = 2; // I know how many it is .................
    //int coresPerSM = 48;  // For Fermi (compute capability 2.x), there are 48 cores per SM
    //int totalCores = numMultiprocessors * coresPerSM;
    //int* data;
    //int* devData;
    //int dataSize = totalCores * sizeof(int);
    //data = (int*)malloc(dataSize);

    //cudaError_t err = myCudaMalloc((void**)devData, dataSize);

    // // Launch the kernel
    //int threadsPerBlock = 32;  // Typically a warp size is 32
    //int numBlocks = totalCores / threadsPerBlock;

    /// <summary>
    /// ///////////////////////////
    /// </summary>

    std::cout << " device count val " << deviceCount << std::endl;

    // Loop through all devices and get properties
    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        if (myCudaGetDeviceProperties(&deviceProp, device) == 0) {
            //std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
            //std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
            //std::cout << "  Multiprocessor Count: " << deviceProp.multiProcessorCount << std::endl;
            //std::cout << "  Core Clock Rate: " << deviceProp.clockRate / 1000 << " MHz" << std::endl;
            //stdd::cout << "  Memory Clock Rate: " << deviceProp.memoryClockRate / 1000 << " MHz" << std::endl;
           //td::cout << "  Memory Bus Width: " << deviceProp.memoryBusWidth << " bits" << std::endl;
            //std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
            std::cout << "Device Name: " << deviceProp.name << std::endl;
            std::cout << "Total Global Memory: " << deviceProp.totalGlobalMem << " bytes" << std::endl;
            std::cout << "Shared Memory Per Block: " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
            std::cout << "Registers Per Block: " << deviceProp.regsPerBlock << std::endl;
            std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
            std::cout << "Max Threads Per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
            std::cout << "Clock Rate: " << deviceProp.clockRate / 1000 << " MHz" << std::endl;
            std::cout << "Multiprocessor Count: " << deviceProp.multiProcessorCount << std::endl;
        }
       
        else {
            std::cerr << "Failed to get properties for device " << device << std::endl;
        }
    }

    // Clean up and unload the library
    FreeLibrary(hCuda);
}
