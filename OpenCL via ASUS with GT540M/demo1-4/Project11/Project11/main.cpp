#include <windows.h>
#include <stdio.h>

// Manually define NVML types
typedef void* nvmlDevice_t;
typedef int nvmlReturn_t;
typedef void* nvmlUnit_t;
typedef void* nvmlVgpuTypeId_t;

// Manually define return codes (check NVML documentation for all values)
#define NVML_SUCCESS 0

// Manually define clock types (from NVML documentation)
typedef enum nvmlClockType_t {
    NVML_CLOCK_GRAPHICS = 0,
    NVML_CLOCK_SM = 1,
    NVML_CLOCK_MEM = 2,
    NVML_CLOCK_VIDEO = 3
} nvmlClockType_t;

typedef enum nvmlClockId_t {
    NVML_CLOCK_ID_CURRENT = 0,
   // Current actual clock value.
    NVML_CLOCK_ID_APP_CLOCK_TARGET = 1,
    //Target application clock.
    NVML_CLOCK_ID_APP_CLOCK_DEFAULT = 2,
   // Default application clock target.
    NVML_CLOCK_ID_CUSTOMER_BOOST_MAX = 3
   // OEM - defined maximum clock rate.
    //NVML_CLOCK_ID_COUNT
    //Count of Clock Ids.
} nvmlClockId_t;

// Manually declare function pointer types
typedef nvmlReturn_t(*nvmlInit_v2_t)(void);
typedef nvmlReturn_t(*nvmlShutdown_t)(void);
typedef nvmlReturn_t(*nvmlDeviceGetHandleByIndex_t)(unsigned int index, nvmlDevice_t* device);
typedef nvmlReturn_t(*nvmlDeviceGetClockInfo_t)(nvmlDevice_t device, nvmlClockType_t type, unsigned int* clockMHz);
typedef const char* (*nvmlErrorString_t)(nvmlReturn_t result);
typedef nvmlReturn_t(*nvmlDeviceGetTemperature_t)(nvmlUnit_t unit, unsigned int  type, unsigned int* temp);
typedef nvmlReturn_t(*nvmlDeviceGetActiveVgpus_t)(nvmlDevice_t device, unsigned int* vgpuCount, nvmlVgpuTypeId_t* vgpuTypeIds);
typedef nvmlReturn_t(*nvmlDeviceGetSupportedGraphicsClocks_t)(nvmlDevice_t device, unsigned int  memoryClockMHz, unsigned int* count, unsigned int* clocksMHz);
typedef nvmlReturn_t(*nvmlDeviceGetApplicationsClock_t)(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int* clockMHz);
typedef nvmlReturn_t(*nvmlDeviceGetClock_t)(nvmlDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int* clockMHz);
//typedef nvmlReturn_t(*nvmlDeviceGetClockInfo_t)(nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock);


int main() {
    // Load the nvml.dll dynamically
    HMODULE hNvml = LoadLibrary(L"C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
    if (hNvml == NULL) {
        printf("Failed to load nvml.dll %d\n", GetLastError());
        return -1;
    }

    // Dynamically load required NVML functions using GetProcAddress
    nvmlInit_v2_t nvmlInit_v2 = (nvmlInit_v2_t)GetProcAddress(hNvml, "nvmlInit_v2");
    nvmlShutdown_t nvmlShutdown = (nvmlShutdown_t)GetProcAddress(hNvml, "nvmlShutdown");
    nvmlDeviceGetHandleByIndex_t nvmlDeviceGetHandleByIndex = (nvmlDeviceGetHandleByIndex_t)GetProcAddress(hNvml, "nvmlDeviceGetHandleByIndex");
    nvmlDeviceGetClockInfo_t nvmlDeviceGetClockInfo = (nvmlDeviceGetClockInfo_t)GetProcAddress(hNvml, "nvmlDeviceGetClockInfo");
    nvmlErrorString_t nvmlErrorString = (nvmlErrorString_t)GetProcAddress(hNvml, "nvmlErrorString");
    nvmlDeviceGetTemperature_t nvmlDeviceGetTemperature = (nvmlDeviceGetTemperature_t)GetProcAddress(hNvml, "nvmlDeviceGetTemperature");
    nvmlDeviceGetActiveVgpus_t nvmlDeviceGetActiveVgpus = (nvmlDeviceGetActiveVgpus_t)GetProcAddress(hNvml, "nvmlDeviceGetActiveVgpus");
    nvmlDeviceGetSupportedGraphicsClocks_t nvmlDeviceGetSupportedGraphicsClocks = (nvmlDeviceGetSupportedGraphicsClocks_t)GetProcAddress(hNvml, "nvmlDeviceGetSupportedGraphicsClocks");
    nvmlDeviceGetApplicationsClock_t nvmlDeviceGetApplicationsClock = (nvmlDeviceGetApplicationsClock_t)GetProcAddress(hNvml, "nvmlDeviceGetApplicationsClock");
    nvmlDeviceGetClock_t nvmlDeviceGetClock = (nvmlDeviceGetClock_t)GetProcAddress(hNvml, "nvmlDeviceGetClock");
    //nvmlDeviceGetClockInfo_t nvmlDeviceGetClockInfo = (nvmlDeviceGetClockInfo_t)GetProcAddress(hNvml, "nvmlDeviceGetClockInfo");

    // Check if the functions were loaded properly
    if (!nvmlInit_v2 || !nvmlShutdown || !nvmlDeviceGetHandleByIndex || !nvmlDeviceGetClockInfo || !nvmlErrorString) {
        printf("Failed to load one or more NVML functions\n");
        FreeLibrary(hNvml);
        return -1;
    }

    // Initialize the NVML library
    nvmlReturn_t result = nvmlInit_v2();
    if (result != NVML_SUCCESS) {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
        FreeLibrary(hNvml);
        return -1;
    }

    // Get a handle to the first GPU (index 0)
    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS) {
        printf("Failed to get device handle: %s\n", nvmlErrorString(result));
        nvmlShutdown();
        FreeLibrary(hNvml);
        return -1;
    }

    // Retrieve the clock info for the GPU core clock
    unsigned int clockMHz;
    result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &clockMHz);
    if (result == NVML_SUCCESS) {
        printf("GPU core clock: %u MHz\n", clockMHz);
    }
    else {
        printf("Failed to get clock info: %s\n", nvmlErrorString(result));
    }

    // get temp
    unsigned int temp;
    result = nvmlDeviceGetTemperature(device, 0, &temp);
    if (result == NVML_SUCCESS) {
        printf("temp %u \n", temp);
    }
    else {
        printf("Failed to get clock info: %s\n", nvmlErrorString(result));
    }

    // demo 1
    nvmlVgpuTypeId_t temp1;
    unsigned int vgpuCount1;
    result = nvmlDeviceGetActiveVgpus(device, &vgpuCount1, &temp1);
    if (result == NVML_SUCCESS) {
        printf("temp 1 %u \n", temp1);
    }
    else {
        printf("Failed to get clock info: %s\n", nvmlErrorString(result));
    }

    //nvmlDeviceGetClock
    unsigned int x5;
    result = nvmlDeviceGetClock(device, NVML_CLOCK_GRAPHICS, NVML_CLOCK_ID_CURRENT, &x5);
    if (result == NVML_SUCCESS) {
        printf("temp 1 %u \n", x5);
    }
    else {
        printf("Failed to get clock info: %s\n", nvmlErrorString(result));
    }

    // nvmlDeviceGetSupportedGraphicsClocks
    unsigned int x1;
    unsigned int x2;
    unsigned int x3;
    result = nvmlDeviceGetSupportedGraphicsClocks(device, x5, &x2, &x3);
    if (result == NVML_SUCCESS) {
        printf("temp 1 %u \n", 1);
    }
    else {
        printf("Failed to get clock info: %s\n", nvmlErrorString(result));
    }

    // nvmlDeviceGetApplicationsClock
    unsigned int x4;
    result = nvmlDeviceGetApplicationsClock(device, NVML_CLOCK_MEM, &x4);
    if (result == NVML_SUCCESS) {
        printf("temp 1 %u \n", x4);
    }
    else {
        printf("Failed to get clock info: %s\n", nvmlErrorString(result));
    }

    // nvmlDeviceGetClockInfo
    unsigned int x6;
    result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS ,&x6);
    if (result == NVML_SUCCESS) {
        printf("temp 1 %u \n", x6);
    }
    else {
        printf("Failed to get clock info: %s\n", nvmlErrorString(result));
    }

    // Shutdown NVML and free the library
    nvmlShutdown();
    FreeLibrary(hNvml);

    return 0;
}



#if 0
#include <Windows.h>
#include <stdio.h>

void __test()
{

}


void __test2()
{

}


int main()
{
	int x = 0;
	while (1) {
		//printf("test");
		//Sleep(1);
		x += 1;
		__test();
		__test2();
	}
	return 0;
}
#endif