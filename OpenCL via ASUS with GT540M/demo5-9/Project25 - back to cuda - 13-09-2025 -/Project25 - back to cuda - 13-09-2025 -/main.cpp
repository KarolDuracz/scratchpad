
/* demo 5 - NIE DZIALA */
#if 0
// cuda_realtime_monitor_fixed.cpp
#define NOMINMAX

#include <windows.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <ctime>    // <--- added for std::time_t, localtime_s

typedef int cudaError_t;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;                // kHz
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int tccDriver;
    int memoryClockRate; // may be 0 in older runtimes
    int memoryBusWidth;
};

// function pointer typedefs (stdcall)
typedef cudaError_t(__stdcall* cudaGetDeviceCount_t)(int*);
typedef cudaError_t(__stdcall* cudaGetDeviceProperties_t)(cudaDeviceProp*, int);
typedef cudaError_t(__stdcall* cudaMemGetInfo_t)(size_t*, size_t*);
typedef cudaError_t(__stdcall* cudaMalloc_t)(void**, size_t);
typedef cudaError_t(__stdcall* cudaFree_t)(void*);
typedef cudaError_t(__stdcall* cudaMemcpy_t)(void*, const void*, size_t, cudaMemcpyKind);
typedef cudaError_t(__stdcall* cudaMemset_t)(void*, int, size_t);
typedef cudaError_t(__stdcall* cudaDeviceSynchronize_t)();
typedef cudaError_t(__stdcall* cudaEventCreate_t)(void**);
typedef cudaError_t(__stdcall* cudaEventRecord_t)(void*, void*);
typedef cudaError_t(__stdcall* cudaEventSynchronize_t)(void*);
typedef cudaError_t(__stdcall* cudaEventElapsedTime_t)(float*, void*, void*);
typedef cudaError_t(__stdcall* cudaEventDestroy_t)(void*);
typedef const char* (__stdcall* cudaGetErrorString_t)(cudaError_t);
typedef cudaError_t(__stdcall* cudaHostAlloc_t)(void**, size_t, unsigned int);
typedef cudaError_t(__stdcall* cudaFreeHost_t)(void*);

// simple mapping
static int coresPerSM(int major, int minor) {
    if (major == 1) return 8;
    if (major == 2) return (minor == 0) ? 32 : 48;
    if (major == 3) return 192;
    if (major == 5) return 128;
    if (major == 6) return 64;
    if (major >= 7) return 64;
    return 64;
}

// Append log line to file
void appendLogLine(const std::string& line) {
    const char* path = "C:\\Windows\\Temp\\log.txt";
    std::ofstream ofs(path, std::ios::app);
    if (!ofs.is_open()) {
        std::cerr << "Unable to open log file: " << path << "\n";
        return;
    }
    ofs << line << "\n";
}

// high-resolution host timing helper (ms)
double hostTimeMs() {
    static LARGE_INTEGER freq = []() { LARGE_INTEGER f; QueryPerformanceFrequency(&f); return f; }();
    LARGE_INTEGER now; QueryPerformanceCounter(&now);
    return (double(now.QuadPart) / double(freq.QuadPart)) * 1000.0;
}

int main() {
    const wchar_t* DllPath = L"C:\\CUDA\\bin\\cudart.dll";
    HMODULE h = LoadLibraryW(DllPath);
    if (!h) {
        std::cerr << "Failed to load " << "C:\\CUDA\\bin\\cudart.dll" << " (GetLastError=" << GetLastError() << ")\n";
        return 1;
    }

    // resolve only the symbols we rely on
    cudaGetDeviceCount_t myGetDeviceCount = (cudaGetDeviceCount_t)GetProcAddress(h, "cudaGetDeviceCount");
    cudaGetDeviceProperties_t myGetDeviceProp = (cudaGetDeviceProperties_t)GetProcAddress(h, "cudaGetDeviceProperties");
    cudaMemGetInfo_t myMemGetInfo = (cudaMemGetInfo_t)GetProcAddress(h, "cudaMemGetInfo");
    cudaMalloc_t myMalloc = (cudaMalloc_t)GetProcAddress(h, "cudaMalloc");
    cudaFree_t myFree = (cudaFree_t)GetProcAddress(h, "cudaFree");
    cudaMemcpy_t myMemcpy = (cudaMemcpy_t)GetProcAddress(h, "cudaMemcpy");
    cudaMemset_t myMemset = (cudaMemset_t)GetProcAddress(h, "cudaMemset");
    cudaDeviceSynchronize_t myDeviceSync = (cudaDeviceSynchronize_t)GetProcAddress(h, "cudaDeviceSynchronize");
    cudaEventCreate_t myEventCreate = (cudaEventCreate_t)GetProcAddress(h, "cudaEventCreate");
    cudaEventRecord_t myEventRecord = (cudaEventRecord_t)GetProcAddress(h, "cudaEventRecord");
    cudaEventSynchronize_t myEventSync = (cudaEventSynchronize_t)GetProcAddress(h, "cudaEventSynchronize");
    cudaEventElapsedTime_t myEventElapsed = (cudaEventElapsedTime_t)GetProcAddress(h, "cudaEventElapsedTime");
    cudaEventDestroy_t myEventDestroy = (cudaEventDestroy_t)GetProcAddress(h, "cudaEventDestroy");
    cudaGetErrorString_t myGetErrorString = (cudaGetErrorString_t)GetProcAddress(h, "cudaGetErrorString");
    cudaHostAlloc_t myHostAlloc = (cudaHostAlloc_t)GetProcAddress(h, "cudaHostAlloc");
    cudaFreeHost_t myFreeHost = (cudaFreeHost_t)GetProcAddress(h, "cudaFreeHost");

    if (!myGetDeviceCount || !myGetDeviceProp || !myMalloc || !myFree || !myMemcpy || !myGetErrorString) {
        std::cerr << "Required cudart symbols missing from DLL. Aborting.\n";
        FreeLibrary(h);
        return 2;
    }

    int deviceCount = 0;
    if (myGetDeviceCount(&deviceCount) != 0 || deviceCount <= 0) {
        std::cerr << "No CUDA devices found or cudaGetDeviceCount failed.\n";
        FreeLibrary(h);
        return 3;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s). Starting realtime monitor (Ctrl-C to stop)\n";

    struct DevContext {
        cudaDeviceProp prop;
        bool hasMemGetInfo;
        bool hasEvents;
        void* devBufA;
        void* devBufB;
        void* hostPinned;
        void* evStart;
        void* evStop;
    };
    std::vector<DevContext> devs(deviceCount);

    // Setup per-device
    const size_t testBytes = 4 * 1024 * 1024; // 4MB
    for (int d = 0; d < deviceCount; ++d) {
        DevContext& C = devs[d];
        memset(&C.prop, 0, sizeof(C.prop));
        if (myGetDeviceProp(&C.prop, d) != 0) {
            std::cerr << "cudaGetDeviceProperties failed for device " << d << "\n";
            continue;
        }
        C.hasMemGetInfo = (myMemGetInfo != nullptr);
        C.hasEvents = (myEventCreate != nullptr && myEventRecord != nullptr && myEventElapsed != nullptr && myEventSync != nullptr && myEventDestroy != nullptr);


        C.devBufA = nullptr; C.devBufB = nullptr; C.hostPinned = nullptr; C.evStart = nullptr; C.evStop = nullptr;
        if (myMalloc(&C.devBufA, testBytes) != 0) C.devBufA = nullptr;
        if (myMalloc(&C.devBufB, testBytes) != 0) C.devBufB = nullptr;
        if (myHostAlloc) {
            if (myHostAlloc(&C.hostPinned, testBytes, 0) != 0) C.hostPinned = nullptr;
        }
        if (!C.hostPinned) {
            C.hostPinned = malloc(testBytes);
        }
        if (C.hostPinned) memset(C.hostPinned, 0xA5, testBytes);

        if (C.hasEvents) {
            if (myEventCreate(&C.evStart) != 0) C.evStart = nullptr;
            if (myEventCreate(&C.evStop) != 0) C.evStop = nullptr;
            if (!C.evStart || !C.evStop) {
                if (C.evStart) myEventDestroy(C.evStart);
                if (C.evStop) myEventDestroy(C.evStop);
                C.evStart = C.evStop = nullptr;
                C.hasEvents = false;
            }
        }
    }

    // Main loop: once per second
    while (true) {
        auto loopStart = std::chrono::steady_clock::now();

        for (int d = 0; d < deviceCount; ++d) {
            DevContext& C = devs[d];
            auto& prop = C.prop;

            int major = prop.major;
            int minor = prop.minor;
            int sm = prop.multiProcessorCount;
            int cores_sm = coresPerSM(major, minor);
            long long total_cores = (long long)sm * cores_sm;

            size_t freeBytes = 0, totalBytes = prop.totalGlobalMem;
            if (C.hasMemGetInfo && myMemGetInfo) {
                myMemGetInfo(&freeBytes, &totalBytes);
            }

            bool memInfoAvailable = (prop.memoryClockRate != 0 && prop.memoryBusWidth != 0);
            double approxMemBW_GB = 0.0;
            if (memInfoAvailable) {
                double memHz = double(prop.memoryClockRate) * 1000.0;
                double bytesPerCycle = double(prop.memoryBusWidth) / 8.0;
                approxMemBW_GB = memHz * bytesPerCycle * 2.0 / 1e9;
            }

            // Build a safe timestamp with localtime_s
            std::time_t tnow = std::time(nullptr);
            std::tm local_tm;
            localtime_s(&local_tm, &tnow);
            std::ostringstream header;
            header << "[" << std::put_time(&local_tm, "%F %T") << "] "
                << "Device " << d << ": " << prop.name
                << " | CC=" << major << "." << minor
                << " | SMs=" << sm << " | cores/SM=" << cores_sm << " | total_cores=" << total_cores
                << " | totalMem=" << (prop.totalGlobalMem / (1024ULL * 1024ULL)) << "MB";
            if (C.hasMemGetInfo) header << " | free=" << (freeBytes / (1024ULL * 1024ULL)) << "MB";
            if (memInfoAvailable) header << " | memClk=" << prop.memoryClockRate << "kHz bus=" << prop.memoryBusWidth << "bit approxBW=" << std::fixed << std::setprecision(2) << approxMemBW_GB << "GB/s";
            std::string hdr = header.str();
            std::cout << hdr << "\n";

            // Micro-tests
            double h2d_ms = -1.0, d2d_ms = -1.0, d2h_ms = -1.0, memset_ms = -1.0, alloc_ms = -1.0, free_ms = -1.0;

            // H2D
            if (C.devBufA && C.hostPinned) {
                double t0 = hostTimeMs();
                cudaError_t r1 = myMemcpy(C.devBufA, C.hostPinned, testBytes, cudaMemcpyHostToDevice);
                double t1 = hostTimeMs();
                if (r1 == 0) h2d_ms = (t1 - t0);
            }

            // D2D
            if (C.devBufA && C.devBufB) {
                if (C.hasEvents && C.evStart && C.evStop && myEventRecord && myEventElapsed && myEventSync) {
                    myEventRecord(C.evStart, 0);
                    cudaError_t r2 = myMemcpy(C.devBufB, C.devBufA, testBytes, cudaMemcpyDeviceToDevice);
                    myEventRecord(C.evStop, 0);
                    myEventSync(C.evStop);
                    float ms = 0.0f;
                    if (myEventElapsed(&ms, C.evStart, C.evStop) == 0) d2d_ms = ms;
                }
                else {
                    double t0 = hostTimeMs();
                    cudaError_t r2 = myMemcpy(C.devBufB, C.devBufA, testBytes, cudaMemcpyDeviceToDevice);
                    double t1 = hostTimeMs();
                    if (r2 == 0) d2d_ms = (t1 - t0);
                }
            }

            // D2H
            if (C.devBufB && C.hostPinned) {
                double t0 = hostTimeMs();
                cudaError_t r3 = myMemcpy(C.hostPinned, C.devBufB, testBytes, cudaMemcpyDeviceToHost);
                double t1 = hostTimeMs();
                if (r3 == 0) d2h_ms = (t1 - t0);
            }

            // Memset on device
            if (C.devBufA) {
                if (C.hasEvents && C.evStart && C.evStop) {
                    myEventRecord(C.evStart, 0);
                    myMemset(C.devBufA, (int)0x42, testBytes);
                    myEventRecord(C.evStop, 0);
                    myEventSync(C.evStop);
                    float ms = 0.0f;
                    if (myEventElapsed(&ms, C.evStart, C.evStop) == 0) memset_ms = ms;
                }
                else {
                    double t0 = hostTimeMs();
                    myMemset(C.devBufA, (int)0x42, testBytes);
                    if (myDeviceSync) myDeviceSync();
                    double t1 = hostTimeMs();
                    memset_ms = t1 - t0;
                }
            }

            // small alloc/free timed
            {
                void* p = nullptr;
                double t0 = hostTimeMs();
                cudaError_t errAlloc = myMalloc(&p, 1024 * 1024); // 1MB
                double t1 = hostTimeMs();
                if (errAlloc == 0 && p) alloc_ms = (t1 - t0);
                double t2 = hostTimeMs();
                if (p) myFree(p);
                double t3 = hostTimeMs();
                if (p) free_ms = (t3 - t2);
            }

            // Build log line safely (use formatted numbers, avoid temporary addresses)
            std::ostringstream oss;
            std::time_t tnow2 = std::time(nullptr);
            std::tm local_tm2;
            localtime_s(&local_tm2, &tnow2);
            oss << "[" << std::put_time(&local_tm2, "%F %T") << "] "
                << "dev=" << d << " name=\"" << prop.name << "\""
                << " h2d_ms=";
            if (h2d_ms >= 0.0) oss << std::fixed << std::setprecision(3) << h2d_ms; else oss << "N/A";
            oss << " d2d_ms=";
            if (d2d_ms >= 0.0) oss << std::fixed << std::setprecision(3) << d2d_ms; else oss << "N/A";
            oss << " d2h_ms=";
            if (d2h_ms >= 0.0) oss << std::fixed << std::setprecision(3) << d2h_ms; else oss << "N/A";
            oss << " memset_ms=";
            if (memset_ms >= 0.0) oss << std::fixed << std::setprecision(3) << memset_ms; else oss << "N/A";
            oss << " alloc_ms=";
            if (alloc_ms >= 0.0) oss << std::fixed << std::setprecision(3) << alloc_ms; else oss << "N/A";
            oss << " free_ms=";
            if (free_ms >= 0.0) oss << std::fixed << std::setprecision(3) << free_ms; else oss << "N/A";

            std::string line = oss.str();
            std::cout << "  " << line << "\n";
            appendLogLine(line);
        }

        // Sleep until 1 second has passed since loopStart
        auto loopEnd = std::chrono::steady_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(loopEnd - loopStart).count();
        if (elapsedMs < 1000) Sleep(DWORD(1000 - elapsedMs));
    }

    // unreachable; cleanup left for completeness
    // (if you add a termination condition, free resources here)
    FreeLibrary(h);
    return 0;
}
#endif


/* demo 4 */
#if 0
// cuda_memory_pressure_test_fixed.cpp
// Dynamically load cudart and run baseline vs memory-pressure bandwidth tests.
// Writes summary lines to C:\Windows\Temp\log.txt
#define NOMINMAX

#include <windows.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <string>
#include <limits>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>      // required for std::ostringstream
#include <functional>   // required for std::function
#include <cstring>      // for memset

typedef int cudaError_t;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int tccDriver;
};

// function pointer typedefs (stdcall)
typedef cudaError_t(__stdcall* cudaMalloc_t)(void**, size_t);
typedef cudaError_t(__stdcall* cudaFree_t)(void*);
typedef cudaError_t(__stdcall* cudaMemcpy_t)(void*, const void*, size_t, cudaMemcpyKind);
typedef const char* (__stdcall* cudaGetErrorString_t)(cudaError_t);
typedef cudaError_t(__stdcall* cudaGetDeviceProperties_t)(cudaDeviceProp*, int);
typedef cudaError_t(__stdcall* cudaGetDeviceCount_t)(int*);
typedef cudaError_t(__stdcall* cudaDeviceSynchronize_t)();
typedef cudaError_t(__stdcall* cudaMemset_t)(void*, int, size_t);
typedef cudaError_t(__stdcall* cudaEventCreate_t)(void**);
typedef cudaError_t(__stdcall* cudaEventRecord_t)(void*, void*);
typedef cudaError_t(__stdcall* cudaEventSynchronize_t)(void*);
typedef cudaError_t(__stdcall* cudaEventElapsedTime_t)(float*, void*, void*);
typedef cudaError_t(__stdcall* cudaEventDestroy_t)(void*);
typedef cudaError_t(__stdcall* cudaHostAlloc_t)(void**, size_t, unsigned int);
typedef cudaError_t(__stdcall* cudaFreeHost_t)(void*);
typedef cudaError_t(__stdcall* cudaMemGetInfo_t)(size_t*, size_t*);

struct CudaRuntime {
    HMODULE h = nullptr;
    cudaMalloc_t cudaMallocFunc = nullptr;
    cudaFree_t cudaFreeFunc = nullptr;
    cudaMemcpy_t cudaMemcpyFunc = nullptr;
    cudaGetErrorString_t cudaGetErrorStringFunc = nullptr;
    cudaGetDeviceProperties_t cudaGetDevicePropertiesFunc = nullptr;
    cudaGetDeviceCount_t cudaGetDeviceCountFunc = nullptr;
    cudaDeviceSynchronize_t cudaDeviceSynchronizeFunc = nullptr;
    cudaMemset_t cudaMemsetFunc = nullptr;
    cudaEventCreate_t cudaEventCreateFunc = nullptr;
    cudaEventRecord_t cudaEventRecordFunc = nullptr;
    cudaEventSynchronize_t cudaEventSynchronizeFunc = nullptr;
    cudaEventElapsedTime_t cudaEventElapsedTimeFunc = nullptr;
    cudaEventDestroy_t cudaEventDestroyFunc = nullptr;
    cudaHostAlloc_t cudaHostAllocFunc = nullptr;
    cudaFreeHost_t cudaFreeHostFunc = nullptr;
    cudaMemGetInfo_t cudaMemGetInfoFunc = nullptr;

    bool load() {
        const wchar_t* candidates[] = {
            L"C:\\CUDA\\bin\\cudart.dll",
            L"cudart64_80.dll",
            L"cudart64_75.dll",
            L"cudart64_72.dll",
            L"cudart64_70.dll",
            L"cudart64_60.dll",
            L"cudart.dll"
        };
        for (auto& name : candidates) {
            HMODULE m = LoadLibraryW(name);
            if (m) { h = m; break; }
        }
        if (!h) return false;

        cudaMallocFunc = (cudaMalloc_t)GetProcAddress(h, "cudaMalloc");
        cudaFreeFunc = (cudaFree_t)GetProcAddress(h, "cudaFree");
        cudaMemcpyFunc = (cudaMemcpy_t)GetProcAddress(h, "cudaMemcpy");
        cudaGetErrorStringFunc = (cudaGetErrorString_t)GetProcAddress(h, "cudaGetErrorString");
        cudaGetDevicePropertiesFunc = (cudaGetDeviceProperties_t)GetProcAddress(h, "cudaGetDeviceProperties");
        cudaGetDeviceCountFunc = (cudaGetDeviceCount_t)GetProcAddress(h, "cudaGetDeviceCount");
        cudaDeviceSynchronizeFunc = (cudaDeviceSynchronize_t)GetProcAddress(h, "cudaDeviceSynchronize");
        cudaMemsetFunc = (cudaMemset_t)GetProcAddress(h, "cudaMemset");
        cudaEventCreateFunc = (cudaEventCreate_t)GetProcAddress(h, "cudaEventCreate");
        cudaEventRecordFunc = (cudaEventRecord_t)GetProcAddress(h, "cudaEventRecord");
        cudaEventSynchronizeFunc = (cudaEventSynchronize_t)GetProcAddress(h, "cudaEventSynchronize");
        cudaEventElapsedTimeFunc = (cudaEventElapsedTime_t)GetProcAddress(h, "cudaEventElapsedTime");
        cudaEventDestroyFunc = (cudaEventDestroy_t)GetProcAddress(h, "cudaEventDestroy");
        cudaHostAllocFunc = (cudaHostAlloc_t)GetProcAddress(h, "cudaHostAlloc");
        cudaFreeHostFunc = (cudaFreeHost_t)GetProcAddress(h, "cudaFreeHost");
        cudaMemGetInfoFunc = (cudaMemGetInfo_t)GetProcAddress(h, "cudaMemGetInfo");

        if (!cudaMallocFunc || !cudaFreeFunc || !cudaMemcpyFunc || !cudaGetErrorStringFunc || !cudaGetDeviceCountFunc || !cudaGetDevicePropertiesFunc) {
            FreeLibrary(h);
            h = nullptr;
            return false;
        }
        return true;
    }

    void unload() {
        if (h) { FreeLibrary(h); h = nullptr; }
    }

    const char* getErrorString(cudaError_t e) const {
        if (cudaGetErrorStringFunc) return cudaGetErrorStringFunc(e);
        return "Unknown";
    }
};

// minimal cores-per-SM mapping (not used heavily here, kept for info)
static int coresPerSM(int major, int minor) {
    if (major == 1) return 8;
    if (major == 2) return (minor == 0) ? 32 : 48;
    if (major == 3) return 192;
    if (major == 5) return 128;
    if (major == 6) return 64;
    if (major >= 7) return 64;
    return 64;
}

static double qpcToSeconds(LARGE_INTEGER start, LARGE_INTEGER end) {
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    return double(end.QuadPart - start.QuadPart) / double(freq.QuadPart);
}

void appendLog(const std::string& line) {
    const char* path = "C:\\Windows\\Temp\\log.txt";
    std::ofstream ofs(path, std::ios::app);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open log file " << path << " for append\n";
        return;
    }
    ofs << line << std::endl;
}

// measure memcpy, try to use events for device timing if available
double measureMemcpyWithEventsOrHost(CudaRuntime& r, void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, bool useEvents) {
    if (useEvents && r.cudaEventCreateFunc && r.cudaEventRecordFunc && r.cudaEventElapsedTimeFunc && r.cudaEventSynchronizeFunc && r.cudaEventDestroyFunc) {
        void* evStart = nullptr;
        void* evStop = nullptr;
        if (r.cudaEventCreateFunc(&evStart) == 0 && r.cudaEventCreateFunc(&evStop) == 0) {
            r.cudaEventRecordFunc(evStart, 0);
            cudaError_t rc = r.cudaMemcpyFunc(dst, src, bytes, kind);
            r.cudaEventRecordFunc(evStop, 0);
            r.cudaEventSynchronizeFunc(evStop);
            float ms = 0.0f;
            if (r.cudaEventElapsedTimeFunc(&ms, evStart, evStop) == 0) {
                r.cudaEventDestroyFunc(evStart);
                r.cudaEventDestroyFunc(evStop);
                return double(ms);
            }
            r.cudaEventDestroyFunc(evStart);
            r.cudaEventDestroyFunc(evStop);
        }
    }
    LARGE_INTEGER t0, t1;
    QueryPerformanceCounter(&t0);
    r.cudaMemcpyFunc(dst, src, bytes, kind);
    QueryPerformanceCounter(&t1);
    return qpcToSeconds(t0, t1) * 1000.0;
}

double computeMBps(size_t bytes, double ms) {
    if (ms <= 0.0) return 0.0;
    double secs = ms / 1000.0;
    return double(bytes) / secs / (1024.0 * 1024.0);
}

int main() {
    CudaRuntime runt;
    if (!runt.load()) {
        std::cerr << "Failed to load cudart DLL or required symbols. Ensure a compatible cudart DLL is available.\n";
        return 1;
    }

    int deviceCount = 0;
    cudaError_t rc = runt.cudaGetDeviceCountFunc(&deviceCount);
    if (rc != 0) {
        std::cerr << "cudaGetDeviceCount failed: " << runt.getErrorString(rc) << "\n";
        runt.unload();
        return 2;
    }
    if (deviceCount <= 0) {
        std::cerr << "No CUDA devices found.\n";
        runt.unload();
        return 3;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop{};
        rc = runt.cudaGetDevicePropertiesFunc(&prop, dev);
        if (rc != 0) {
            std::cerr << "Failed to get properties for device " << dev << ": " << runt.getErrorString(rc) << "\n";
            continue;
        }

        std::string devName(prop.name);
        std::cout << "Device " << dev << ": " << devName << "\n";
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Multiprocessors   : " << prop.multiProcessorCount << "\n";
        std::cout << "  Total Global Mem  : " << prop.totalGlobalMem << " bytes (" << (prop.totalGlobalMem / (1024ULL * 1024ULL)) << " MB)\n";

        // ask cudaMemGetInfo if available for free mem snapshot
        size_t freeBytes = 0, totalBytes = prop.totalGlobalMem;
        if (runt.cudaMemGetInfoFunc) {
            runt.cudaMemGetInfoFunc(&freeBytes, &totalBytes);
            std::cout << "  cudaMemGetInfo => free: " << freeBytes << " total: " << totalBytes << "\n";
        }

        // Choose a buffer size for bandwidth tests.
        size_t targetBuf = std::max<size_t>(1 * 1024 * 1024, std::min<size_t>(prop.totalGlobalMem / 8, 256ULL * 1024ULL * 1024ULL));
        size_t reserve = std::min<size_t>(64ULL * 1024ULL * 1024ULL, prop.totalGlobalMem / 20);
        if (runt.cudaMemGetInfoFunc && freeBytes > reserve + targetBuf) {
            // ok
        }
        else if (runt.cudaMemGetInfoFunc && freeBytes > reserve) {
            if (freeBytes > reserve) targetBuf = freeBytes - reserve;
            if (targetBuf == 0) targetBuf = 1 * 1024 * 1024;
        }

        std::cout << "  Using test buffer size: " << (targetBuf / (1024ULL * 1024ULL)) << " MB\n";

        // prepare host buffer (prefer pinned if available)
        void* hostPinned = nullptr;
        bool usingPinned = false;
        std::vector<char> hostPageable;
        if (runt.cudaHostAllocFunc) {
            if (runt.cudaHostAllocFunc(&hostPinned, targetBuf, 0) == 0 && hostPinned) {
                usingPinned = true;
                std::memset(hostPinned, 0xA5, targetBuf);
            }
            else {
                hostPinned = nullptr;
            }
        }
        if (!usingPinned) {
            hostPageable.resize(targetBuf);
            std::fill(hostPageable.begin(), hostPageable.end(), 0xA5);
            hostPinned = hostPageable.data();
        }

        // allocate two large device buffers
        void* dA = nullptr; void* dB = nullptr;
        rc = runt.cudaMallocFunc(&dA, targetBuf);
        if (rc != 0 || dA == nullptr) {
            std::cerr << "  Failed to allocate device buffer A: " << runt.getErrorString(rc) << "\n";
            if (usingPinned && runt.cudaFreeHostFunc) runt.cudaFreeHostFunc(hostPinned);
            continue;
        }
        rc = runt.cudaMallocFunc(&dB, targetBuf);
        if (rc != 0 || dB == nullptr) {
            std::cerr << "  Failed to allocate device buffer B: " << runt.getErrorString(rc) << "\n";
            runt.cudaFreeFunc(dA);
            if (usingPinned && runt.cudaFreeHostFunc) runt.cudaFreeHostFunc(hostPinned);
            continue;
        }

        auto runBandwidth = [&](const char* label, std::function<double()> runner, int iterations = 10) -> double {
            double sum_ms = 0.0;
            int good = 0;
            for (int i = 0; i < iterations; ++i) {
                double ms = runner();
                if (ms > 0.0) { sum_ms += ms; good++; }
            }
            double avg_ms = (good > 0) ? (sum_ms / double(good)) : 0.0;
            double mbps = computeMBps(targetBuf, avg_ms);
            std::ostringstream oss;
            oss << label << " avg_ms=" << std::fixed << std::setprecision(3) << avg_ms << " MB/s=" << std::fixed << std::setprecision(2) << mbps;
            std::string line = oss.str();
            appendLog(line);
            std::cout << "    " << line << "\n";
            return mbps;
        };

        std::cout << "Running baseline bandwidth tests...\n";
        bool haveEvents = (runt.cudaEventCreateFunc && runt.cudaEventRecordFunc && runt.cudaEventElapsedTimeFunc && runt.cudaEventSynchronizeFunc && runt.cudaEventDestroyFunc);

        if (runt.cudaDeviceSynchronizeFunc) runt.cudaDeviceSynchronizeFunc();

        double h2d_mb = runBandwidth("Baseline H2D (host->device)", [&]() {
            return measureMemcpyWithEventsOrHost(runt, dA, hostPinned, targetBuf, cudaMemcpyHostToDevice, haveEvents);
            });

        double d2h_mb = runBandwidth("Baseline D2H (device->host)", [&]() {
            runt.cudaMemcpyFunc(dA, hostPinned, targetBuf, cudaMemcpyHostToDevice);
            return measureMemcpyWithEventsOrHost(runt, hostPinned, dA, targetBuf, cudaMemcpyDeviceToHost, haveEvents);
            });

        double d2d_mb = runBandwidth("Baseline D2D (device->device)", [&]() {
            runt.cudaMemcpyFunc(dA, hostPinned, targetBuf, cudaMemcpyHostToDevice);
            return measureMemcpyWithEventsOrHost(runt, dB, dA, targetBuf, cudaMemcpyDeviceToDevice, haveEvents);
            });

        // Now create memory pressure
        std::cout << "\nCreating memory pressure: allocating chunks until near-full\n";
        size_t totalMem = prop.totalGlobalMem;
        size_t reserved = std::max<size_t>(64ULL * 1024ULL * 1024ULL, totalMem / 40);
        size_t targetUsed = (totalMem > reserved) ? (totalMem - reserved) : (totalMem / 2);
        targetUsed = (targetUsed > 2 * targetBuf) ? (targetUsed - 2 * targetBuf) : 0;
        std::vector<void*> pressureAlloc;
        size_t chunk = 4 * 1024 * 1024;
        if (chunk > targetBuf) chunk = targetBuf;

        size_t used = 0;
        while (used < targetUsed) {
            void* p = nullptr;
            cudaError_t err = runt.cudaMallocFunc(&p, chunk);
            if (err == 0 && p != nullptr) {
                pressureAlloc.push_back(p);
                used += chunk;
            }
            else {
                if (chunk > 1024 * 1024) chunk /= 2;
                else break;
            }
        }
        std::cout << "  Allocated pressure blocks: " << pressureAlloc.size() << " total bytes ~ " << used << "\n";
        {
            std::ostringstream ss;
            ss << "Pressure alloc count=" << pressureAlloc.size() << " bytes=" << used;
            appendLog(ss.str());
        }

        // Re-run bandwidth tests under pressure
        std::cout << "\nRunning bandwidth tests under memory pressure...\n";

        double h2d_mb_pressure = runBandwidth("Pressure H2D (host->device)", [&]() {
            return measureMemcpyWithEventsOrHost(runt, dA, hostPinned, targetBuf, cudaMemcpyHostToDevice, haveEvents);
            });

        double d2h_mb_pressure = runBandwidth("Pressure D2H (device->host)", [&]() {
            runt.cudaMemcpyFunc(dA, hostPinned, targetBuf, cudaMemcpyHostToDevice);
            return measureMemcpyWithEventsOrHost(runt, hostPinned, dA, targetBuf, cudaMemcpyDeviceToHost, haveEvents);
        });

        double d2d_mb_pressure = runBandwidth("Pressure D2D (device->device)", [&]() {
            runt.cudaMemcpyFunc(dA, hostPinned, targetBuf, cudaMemcpyHostToDevice);
            return measureMemcpyWithEventsOrHost(runt, dB, dA, targetBuf, cudaMemcpyDeviceToDevice, haveEvents);
            });

        // Summarize
        {
            std::ostringstream finalss;
            finalss << "SUMMARY dev=" << dev << " (" << devName << ") bufMB=" << (targetBuf / (1024.0 * 1024.0))
                << " Baseline H2D=" << h2d_mb << " MB/s D2H=" << d2h_mb << " MB/s D2D=" << d2d_mb
                << " | Pressure H2D=" << h2d_mb_pressure << " MB/s D2H=" << d2h_mb_pressure << " MB/s D2D=" << d2d_mb_pressure;
            std::string finalLine = finalss.str();
            appendLog(finalLine);
            std::cout << "\n" << finalLine << "\n\n";
        }

        // cleanup pressure allocations
        for (void* p : pressureAlloc) {
            runt.cudaFreeFunc(p);
        }
        pressureAlloc.clear();

        // cleanup buffers and host pinned
        runt.cudaFreeFunc(dA);
        runt.cudaFreeFunc(dB);
        if (usingPinned && runt.cudaFreeHostFunc) runt.cudaFreeHostFunc(hostPinned);

        std::cout << "Completed device " << dev << "\n\n-----------------------------------------\n\n";
    }

    runt.unload();
    return 0;
}
#endif


/* demo 3*/
#if 0
// cuda_probe_longrun_fixed.cpp
#define NOMINMAX

#include <windows.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <string>
#include <limits>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <sstream>   // <<< ADDED to fix undefined std::ostringstream and operator<< errors

typedef int cudaError_t; // runtime uses int-like returns

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int tccDriver;
};

// function pointer typedefs (stdcall)
typedef cudaError_t(__stdcall* cudaMalloc_t)(void**, size_t);
typedef cudaError_t(__stdcall* cudaFree_t)(void*);
typedef cudaError_t(__stdcall* cudaMemcpy_t)(void*, const void*, size_t, cudaMemcpyKind);
typedef const char* (__stdcall* cudaGetErrorString_t)(cudaError_t);
typedef cudaError_t(__stdcall* cudaGetDeviceProperties_t)(cudaDeviceProp*, int);
typedef cudaError_t(__stdcall* cudaGetDeviceCount_t)(int*);
typedef cudaError_t(__stdcall* cudaDeviceSynchronize_t)();
typedef cudaError_t(__stdcall* cudaMemset_t)(void*, int, size_t);
typedef cudaError_t(__stdcall* cudaEventCreate_t)(void**);
typedef cudaError_t(__stdcall* cudaEventRecord_t)(void*, void*);
typedef cudaError_t(__stdcall* cudaEventSynchronize_t)(void*);
typedef cudaError_t(__stdcall* cudaEventElapsedTime_t)(float*, void*, void*);
typedef cudaError_t(__stdcall* cudaEventDestroy_t)(void*);
typedef cudaError_t(__stdcall* cudaHostAlloc_t)(void**, size_t, unsigned int);
typedef cudaError_t(__stdcall* cudaFreeHost_t)(void*);
typedef cudaError_t(__stdcall* cudaMemGetInfo_t)(size_t*, size_t*);

struct CudaRuntime {
    HMODULE h = nullptr;
    cudaMalloc_t cudaMallocFunc = nullptr;
    cudaFree_t cudaFreeFunc = nullptr;
    cudaMemcpy_t cudaMemcpyFunc = nullptr;
    cudaGetErrorString_t cudaGetErrorStringFunc = nullptr;
    cudaGetDeviceProperties_t cudaGetDevicePropertiesFunc = nullptr;
    cudaGetDeviceCount_t cudaGetDeviceCountFunc = nullptr;
    cudaDeviceSynchronize_t cudaDeviceSynchronizeFunc = nullptr;
    cudaMemset_t cudaMemsetFunc = nullptr;
    cudaEventCreate_t cudaEventCreateFunc = nullptr;
    cudaEventRecord_t cudaEventRecordFunc = nullptr;
    cudaEventSynchronize_t cudaEventSynchronizeFunc = nullptr;
    cudaEventElapsedTime_t cudaEventElapsedTimeFunc = nullptr;
    cudaEventDestroy_t cudaEventDestroyFunc = nullptr;
    cudaHostAlloc_t cudaHostAllocFunc = nullptr;
    cudaFreeHost_t cudaFreeHostFunc = nullptr;
    cudaMemGetInfo_t cudaMemGetInfoFunc = nullptr;

    bool load() {
        const wchar_t* candidates[] = {
            L"C:\\CUDA\\bin\\cudart.dll",
            L"cudart64_80.dll",
            L"cudart64_75.dll",
            L"cudart64_72.dll",
            L"cudart64_70.dll",
            L"cudart64_60.dll",
            L"cudart.dll"
        };
        for (auto& name : candidates) {
            HMODULE m = LoadLibraryW(name);
            if (m) { h = m; break; }
        }
        if (!h) return false;

        cudaMallocFunc = (cudaMalloc_t)GetProcAddress(h, "cudaMalloc");
        cudaFreeFunc = (cudaFree_t)GetProcAddress(h, "cudaFree");
        cudaMemcpyFunc = (cudaMemcpy_t)GetProcAddress(h, "cudaMemcpy");
        cudaGetErrorStringFunc = (cudaGetErrorString_t)GetProcAddress(h, "cudaGetErrorString");
        cudaGetDevicePropertiesFunc = (cudaGetDeviceProperties_t)GetProcAddress(h, "cudaGetDeviceProperties");
        cudaGetDeviceCountFunc = (cudaGetDeviceCount_t)GetProcAddress(h, "cudaGetDeviceCount");
        cudaDeviceSynchronizeFunc = (cudaDeviceSynchronize_t)GetProcAddress(h, "cudaDeviceSynchronize");
        cudaMemsetFunc = (cudaMemset_t)GetProcAddress(h, "cudaMemset");
        cudaEventCreateFunc = (cudaEventCreate_t)GetProcAddress(h, "cudaEventCreate");
        cudaEventRecordFunc = (cudaEventRecord_t)GetProcAddress(h, "cudaEventRecord");
        cudaEventSynchronizeFunc = (cudaEventSynchronize_t)GetProcAddress(h, "cudaEventSynchronize");
        cudaEventElapsedTimeFunc = (cudaEventElapsedTime_t)GetProcAddress(h, "cudaEventElapsedTime");
        cudaEventDestroyFunc = (cudaEventDestroy_t)GetProcAddress(h, "cudaEventDestroy");
        cudaHostAllocFunc = (cudaHostAlloc_t)GetProcAddress(h, "cudaHostAlloc");
        cudaFreeHostFunc = (cudaFreeHost_t)GetProcAddress(h, "cudaFreeHost");
        cudaMemGetInfoFunc = (cudaMemGetInfo_t)GetProcAddress(h, "cudaMemGetInfo");

        // essential
        if (!cudaMallocFunc || !cudaFreeFunc || !cudaMemcpyFunc || !cudaGetErrorStringFunc || !cudaGetDeviceCountFunc || !cudaGetDevicePropertiesFunc) {
            FreeLibrary(h);
            h = nullptr;
            return false;
        }
        return true;
    }

    void unload() {
        if (h) { FreeLibrary(h); h = nullptr; }
    }

    const char* getErrorString(cudaError_t e) const {
        if (cudaGetErrorStringFunc) return cudaGetErrorStringFunc(e);
        return "Unknown";
    }
};

// mapping cores per SM
static int coresPerSM(int major, int minor) {
    if (major == 1) return 8;
    if (major == 2) return (minor == 0) ? 32 : 48;
    if (major == 3) return 192;
    if (major == 5) return 128;
    if (major == 6) return 64;
    if (major >= 7) return 64;
    return 64;
}

// timing helpers
static double qpcNowSeconds() {
    LARGE_INTEGER freq, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    return double(now.QuadPart) / double(freq.QuadPart);
}
static double qpcToSeconds(LARGE_INTEGER start, LARGE_INTEGER end) {
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    return double(end.QuadPart - start.QuadPart) / double(freq.QuadPart);
}

bool tryCudaMallocTimed(CudaRuntime& r, void** p, size_t bytes, double& elapsedSec, cudaError_t& errOut) {
    LARGE_INTEGER t0, t1;
    QueryPerformanceCounter(&t0);
    cudaError_t rc = r.cudaMallocFunc(p, bytes);
    QueryPerformanceCounter(&t1);
    elapsedSec = qpcToSeconds(t0, t1);
    errOut = rc;
    return (rc == 0 && *p != nullptr);
}

// append a log line to C:\Windows\Temp\log.txt
void appendLog(const std::string& line) {
    const char* path = "C:\\Windows\\Temp\\log.txt";
    std::ofstream ofs(path, std::ios::app);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open log file " << path << " for append\n";
        return;
    }
    ofs << line << std::endl;
    ofs.close();
}

int main() {
    CudaRuntime runt;
    if (!runt.load()) {
        std::cerr << "Failed to load cudart DLL or required symbols. Ensure a compatible cudart DLL is available.\n";
        return 1;
    }

    int deviceCount = 0;
    cudaError_t rc = runt.cudaGetDeviceCountFunc(&deviceCount);
    if (rc != 0) {
        std::cerr << "cudaGetDeviceCount failed: " << runt.getErrorString(rc) << " (" << rc << ")\n";
        runt.unload();
        return 2;
    }
    if (deviceCount <= 0) {
        std::cerr << "No CUDA devices found.\n";
        runt.unload();
        return 3;
    }

    std::cout << "CUDA device count: " << deviceCount << "\n\n";

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop{};
        rc = runt.cudaGetDevicePropertiesFunc(&prop, dev);
        if (rc != 0) {
            std::cerr << "Failed to get properties for device " << dev << ": " << runt.getErrorString(rc) << "\n";
            continue;
        }

        std::string devName(prop.name);
        int major = prop.major;
        int minor = prop.minor;
        int sm = prop.multiProcessorCount;
        int cores_sm = coresPerSM(major, minor);
        long long total_cores = static_cast<long long>(sm) * cores_sm;

        std::cout << "Device " << dev << ": " << devName << "\n";
        std::cout << "  Compute capability : " << major << "." << minor << "\n";
        std::cout << "  Multiprocessors    : " << sm << "\n";
        std::cout << "  Cores per SM       : " << cores_sm << "\n";
        std::cout << "  Total CUDA cores   : " << total_cores << "\n";
        std::cout << "  Total Global Mem   : " << prop.totalGlobalMem << " bytes (" << (prop.totalGlobalMem / (1024ULL * 1024ULL)) << " MB)\n";

        const size_t transferBytes = 4 * 1024 * 1024; // 4MB
        std::vector<char> hostBuffer(transferBytes, 0xA5);

        void* devBufA = nullptr;
        void* devBufB = nullptr;
        rc = runt.cudaMallocFunc(&devBufA, transferBytes);
        if (rc != 0 || devBufA == nullptr) {
            std::cerr << "  Failed to cudaMalloc devBufA (" << runt.getErrorString(rc) << ")\n";
            continue;
        }
        rc = runt.cudaMallocFunc(&devBufB, transferBytes);
        if (rc != 0 || devBufB == nullptr) {
            std::cerr << "  Failed to cudaMalloc devBufB (" << runt.getErrorString(rc) << ")\n";
            runt.cudaFreeFunc(devBufA);
            continue;
        }

        // events if available
        void* evStart = nullptr;
        void* evStop = nullptr;
        bool haveEvents = false;
        if (runt.cudaEventCreateFunc && runt.cudaEventRecordFunc && runt.cudaEventElapsedTimeFunc && runt.cudaEventSynchronizeFunc && runt.cudaEventDestroyFunc) {
            if (runt.cudaEventCreateFunc(&evStart) == 0 && runt.cudaEventCreateFunc(&evStop) == 0) {
                haveEvents = true;
            }
            else {
                if (evStart) runt.cudaEventDestroyFunc(evStart);
                if (evStop) runt.cudaEventDestroyFunc(evStop);
                evStart = evStop = nullptr;
                haveEvents = false;
            }
        }

        const int durations[] = { 10, 20, 30 };
        for (int durIdx = 0; durIdx < (int)(sizeof(durations) / sizeof(durations[0])); ++durIdx) {
            int durationSec = durations[durIdx];
            std::cout << "\nStarting long-run test for " << durationSec << " seconds (transfer " << (transferBytes / 1024 / 1024) << " MB) on device " << dev << "\n";

            size_t h2d_count = 0, d2d_count = 0, d2h_count = 0;
            double h2d_sum_ms = 0.0, d2d_sum_ms = 0.0, d2h_sum_ms = 0.0;

            double testStart = qpcNowSeconds();
            double lastLogTime = testStart;
            double now = testStart;

            while ((now - testStart) < double(durationSec)) {
                // H2D
                LARGE_INTEGER t0, t1;
                QueryPerformanceCounter(&t0);
                rc = runt.cudaMemcpyFunc(devBufA, hostBuffer.data(), transferBytes, cudaMemcpyHostToDevice);
                QueryPerformanceCounter(&t1);
                double h2d_ms = qpcToSeconds(t0, t1) * 1000.0;
                if (rc == 0) { h2d_count++; h2d_sum_ms += h2d_ms; }

                // D2D
                double d2d_ms = -1.0;
                if (haveEvents && evStart && evStop) {
                    runt.cudaEventRecordFunc(evStart, 0);
                    rc = runt.cudaMemcpyFunc(devBufB, devBufA, transferBytes, cudaMemcpyDeviceToDevice);
                    runt.cudaEventRecordFunc(evStop, 0);
                    runt.cudaEventSynchronizeFunc(evStop);
                    float ms = 0.0f;
                    if (runt.cudaEventElapsedTimeFunc(&ms, evStart, evStop) == 0) d2d_ms = ms;
                }
                else {
                    QueryPerformanceCounter(&t0);
                    rc = runt.cudaMemcpyFunc(devBufB, devBufA, transferBytes, cudaMemcpyDeviceToDevice);
                    QueryPerformanceCounter(&t1);
                    d2d_ms = qpcToSeconds(t0, t1) * 1000.0;
                }
                if (rc == 0 && d2d_ms >= 0.0) { d2d_count++; d2d_sum_ms += d2d_ms; }

                // D2H
                QueryPerformanceCounter(&t0);
                rc = runt.cudaMemcpyFunc(hostBuffer.data(), devBufB, transferBytes, cudaMemcpyDeviceToHost);
                QueryPerformanceCounter(&t1);
                double d2h_ms = qpcToSeconds(t0, t1) * 1000.0;
                if (rc == 0) { d2h_count++; d2h_sum_ms += d2h_ms; }

                now = qpcNowSeconds();
                if ((now - lastLogTime) >= 1.0) {
                    double avg_h2d = (h2d_count > 0) ? (h2d_sum_ms / double(h2d_count)) : 0.0;
                    double avg_d2d = (d2d_count > 0) ? (d2d_sum_ms / double(d2d_count)) : 0.0;
                    double avg_d2h = (d2h_count > 0) ? (d2h_sum_ms / double(d2h_count)) : 0.0;

                    std::time_t t = std::time(nullptr);
                    std::tm local_tm;
                    localtime_s(&local_tm, &t);
                    char timebuf[64];
                    std::strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", &local_tm);

                    std::ostringstream oss;
                    oss << timebuf << " | dev=" << dev << " (" << devName << ")"
                        << " | testDur=" << durationSec << "s"
                        << " | transferMB=" << (transferBytes / (1024.0 * 1024.0))
                        << " | H2D_cnt=" << h2d_count << " avg_ms=" << std::fixed << std::setprecision(3) << avg_h2d
                        << " | D2D_cnt=" << d2d_count << " avg_ms=" << avg_d2d
                        << " | D2H_cnt=" << d2h_count << " avg_ms=" << avg_d2h;

                    std::string line = oss.str();
                    appendLog(line);
                    std::cout << line << std::endl;

                    // reset counters for next second
                    h2d_count = d2d_count = d2h_count = 0;
                    h2d_sum_ms = d2d_sum_ms = d2h_sum_ms = 0.0;
                    lastLogTime = now;
                }

                now = qpcNowSeconds();
            } // while duration

            std::cout << "Completed long-run test for " << durationSec << " seconds.\n";
        } // durations loop

        if (evStart && runt.cudaEventDestroyFunc) runt.cudaEventDestroyFunc(evStart);
        if (evStop && runt.cudaEventDestroyFunc) runt.cudaEventDestroyFunc(evStop);

        runt.cudaFreeFunc(devBufA);
        runt.cudaFreeFunc(devBufB);

        std::cout << "\n-------------------------------------------------------------\n\n";
    } // per-device

    runt.unload();
    return 0;
}
#endif


/* demo 2*/
#if 0
// cuda_probe_tests.cpp
// Enhanced dynamic cudart probe + memory/timing tests for older runtimes (Windows).
#define NOMINMAX

#include <windows.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <string>
#include <limits>
#include <chrono>
#include <cmath>

typedef int cudaError_t; // runtime uses int-like returns

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int tccDriver;
};

// basic function pointer typedefs (stdcall)
typedef cudaError_t(__stdcall* cudaMalloc_t)(void**, size_t);
typedef cudaError_t(__stdcall* cudaFree_t)(void*);
typedef cudaError_t(__stdcall* cudaMemcpy_t)(void*, const void*, size_t, cudaMemcpyKind);
typedef const char* (__stdcall* cudaGetErrorString_t)(cudaError_t);
typedef cudaError_t(__stdcall* cudaGetDeviceProperties_t)(cudaDeviceProp*, int);
typedef cudaError_t(__stdcall* cudaGetDeviceCount_t)(int*);
typedef cudaError_t(__stdcall* cudaDeviceSynchronize_t)();
typedef cudaError_t(__stdcall* cudaMemset_t)(void*, int, size_t);
typedef cudaError_t(__stdcall* cudaEventCreate_t)(void**);
typedef cudaError_t(__stdcall* cudaEventRecord_t)(void*, void*);
typedef cudaError_t(__stdcall* cudaEventSynchronize_t)(void*);
typedef cudaError_t(__stdcall* cudaEventElapsedTime_t)(float*, void*, void*);
typedef cudaError_t(__stdcall* cudaEventDestroy_t)(void*);
typedef cudaError_t(__stdcall* cudaHostAlloc_t)(void**, size_t, unsigned int);
typedef cudaError_t(__stdcall* cudaFreeHost_t)(void*);
typedef cudaError_t(__stdcall* cudaMemGetInfo_t)(size_t*, size_t*);

struct CudaRuntime {
    HMODULE h = nullptr;
    cudaMalloc_t cudaMallocFunc = nullptr;
    cudaFree_t cudaFreeFunc = nullptr;
    cudaMemcpy_t cudaMemcpyFunc = nullptr;
    cudaGetErrorString_t cudaGetErrorStringFunc = nullptr;
    cudaGetDeviceProperties_t cudaGetDevicePropertiesFunc = nullptr;
    cudaGetDeviceCount_t cudaGetDeviceCountFunc = nullptr;
    cudaDeviceSynchronize_t cudaDeviceSynchronizeFunc = nullptr;
    cudaMemset_t cudaMemsetFunc = nullptr;
    cudaEventCreate_t cudaEventCreateFunc = nullptr;
    cudaEventRecord_t cudaEventRecordFunc = nullptr;
    cudaEventSynchronize_t cudaEventSynchronizeFunc = nullptr;
    cudaEventElapsedTime_t cudaEventElapsedTimeFunc = nullptr;
    cudaEventDestroy_t cudaEventDestroyFunc = nullptr;
    cudaHostAlloc_t cudaHostAllocFunc = nullptr;
    cudaFreeHost_t cudaFreeHostFunc = nullptr;
    cudaMemGetInfo_t cudaMemGetInfoFunc = nullptr;

    bool load() {
        // load from a few candidate names, prefer local installation path previously used
        const wchar_t* candidates[] = {
            L"C:\\CUDA\\bin\\cudart.dll", // user's prior code
            L"cudart64_80.dll",
            L"cudart64_75.dll",
            L"cudart64_72.dll",
            L"cudart64_70.dll",
            L"cudart64_60.dll",
            L"cudart.dll"
        };
        for (auto& name : candidates) {
            HMODULE m = LoadLibraryW(name);
            if (m) {
                h = m;
                break;
            }
        }
        if (!h) return false;

        cudaMallocFunc = (cudaMalloc_t)GetProcAddress(h, "cudaMalloc");
        cudaFreeFunc = (cudaFree_t)GetProcAddress(h, "cudaFree");
        cudaMemcpyFunc = (cudaMemcpy_t)GetProcAddress(h, "cudaMemcpy");
        cudaGetErrorStringFunc = (cudaGetErrorString_t)GetProcAddress(h, "cudaGetErrorString");
        cudaGetDevicePropertiesFunc = (cudaGetDeviceProperties_t)GetProcAddress(h, "cudaGetDeviceProperties");
        cudaGetDeviceCountFunc = (cudaGetDeviceCount_t)GetProcAddress(h, "cudaGetDeviceCount");
        cudaDeviceSynchronizeFunc = (cudaDeviceSynchronize_t)GetProcAddress(h, "cudaDeviceSynchronize");
        cudaMemsetFunc = (cudaMemset_t)GetProcAddress(h, "cudaMemset");
        cudaEventCreateFunc = (cudaEventCreate_t)GetProcAddress(h, "cudaEventCreate");
        cudaEventRecordFunc = (cudaEventRecord_t)GetProcAddress(h, "cudaEventRecord");
        cudaEventSynchronizeFunc = (cudaEventSynchronize_t)GetProcAddress(h, "cudaEventSynchronize");
        cudaEventElapsedTimeFunc = (cudaEventElapsedTime_t)GetProcAddress(h, "cudaEventElapsedTime");
        cudaEventDestroyFunc = (cudaEventDestroy_t)GetProcAddress(h, "cudaEventDestroy");
        cudaHostAllocFunc = (cudaHostAlloc_t)GetProcAddress(h, "cudaHostAlloc");
        cudaFreeHostFunc = (cudaFreeHost_t)GetProcAddress(h, "cudaFreeHost");
        cudaMemGetInfoFunc = (cudaMemGetInfo_t)GetProcAddress(h, "cudaMemGetInfo");

        // essential functions must exist
        if (!cudaMallocFunc || !cudaFreeFunc || !cudaMemcpyFunc || !cudaGetErrorStringFunc || !cudaGetDeviceCountFunc || !cudaGetDevicePropertiesFunc) {
            FreeLibrary(h);
            h = nullptr;
            return false;
        }
        return true;
    }

    void unload() {
        if (h) {
            FreeLibrary(h);
            h = nullptr;
        }
    }

    const char* getErrorString(cudaError_t e) const {
        if (cudaGetErrorStringFunc) return cudaGetErrorStringFunc(e);
        return "Unknown";
    }
};

// mapping as before (cores per SM)
static int coresPerSM(int major, int minor) {
    if (major == 1) return 8;
    if (major == 2) {
        if (minor == 0) return 32;
        return 48;
    }
    if (major == 3) return 192;
    if (major == 5) return 128;
    if (major == 6) return 64;
    if (major >= 7) return 64;
    return 64;
}

// helpers for host timing
static double qpcSeconds() {
    LARGE_INTEGER freq, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    return double(now.QuadPart) / double(freq.QuadPart);
}
static double qpcToSeconds(LARGE_INTEGER start, LARGE_INTEGER end) {
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    return double(end.QuadPart - start.QuadPart) / double(freq.QuadPart);
}

// safe wrapper for cudaMalloc timing (host)
bool tryCudaMallocTimed(CudaRuntime& r, void** p, size_t bytes, double& elapsedSec, cudaError_t& errOut) {
    LARGE_INTEGER t0, t1;
    QueryPerformanceCounter(&t0);
    cudaError_t rc = r.cudaMallocFunc(p, bytes);
    QueryPerformanceCounter(&t1);
    elapsedSec = qpcToSeconds(t0, t1);
    errOut = rc;
    return (rc == 0 && *p != nullptr);
}

int main() {
    CudaRuntime runt;
    if (!runt.load()) {
        std::cerr << "Failed to load cudart DLL or required symbols. Ensure a compatible cudart DLL is available.\n";
        return 1;
    }

    // device count
    int deviceCount = 0;
    cudaError_t rc = runt.cudaGetDeviceCountFunc(&deviceCount);
    if (rc != 0) {
        std::cerr << "cudaGetDeviceCount failed: " << runt.getErrorString(rc) << " (" << rc << ")\n";
        runt.unload();
        return 2;
    }
    if (deviceCount <= 0) {
        std::cerr << "No CUDA devices found.\n";
        runt.unload();
        return 3;
    }
    std::cout << "CUDA device count: " << deviceCount << "\n\n";

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop{};
        rc = runt.cudaGetDevicePropertiesFunc(&prop, dev);
        if (rc != 0) {
            std::cerr << "Failed to get properties for device " << dev << ": " << runt.getErrorString(rc) << "\n";
            continue;
        }

        std::string devName(prop.name);
        int major = prop.major;
        int minor = prop.minor;
        int sm = prop.multiProcessorCount;
        int cores_sm = coresPerSM(major, minor);
        long long total_cores = static_cast<long long>(sm) * cores_sm;

        std::cout << "Device " << dev << ": " << devName << "\n";
        std::cout << "  Compute capability : " << major << "." << minor << "\n";
        std::cout << "  Multiprocessors    : " << sm << "\n";
        std::cout << "  Cores per SM       : " << cores_sm << "\n";
        std::cout << "  Total CUDA cores   : " << total_cores << "\n";
        std::cout << "  Total Global Mem   : " << prop.totalGlobalMem << " bytes (" << (prop.totalGlobalMem / (1024ULL * 1024ULL)) << " MB)\n";
        std::cout << "  Shared mem / block : " << prop.sharedMemPerBlock << " bytes\n";
        std::cout << "  Warp size          : " << prop.warpSize << "\n";
        std::cout << "  Max threads / block: " << prop.maxThreadsPerBlock << "\n";

        // memory info if available
        if (runt.cudaMemGetInfoFunc) {
            size_t freeBytes = 0, totalBytes = 0;
            runt.cudaMemGetInfoFunc(&freeBytes, &totalBytes);
            std::cout << "  cudaMemGetInfo => free: " << freeBytes << " bytes, total: " << totalBytes << " bytes\n";
        }

        // 1) Single-allocation max probe (binary-style)
        size_t totalMem = prop.totalGlobalMem;
        size_t upper = (totalMem > (1ULL << 20)) ? (totalMem - (1ULL << 20)) : (totalMem / 2); // leave ~1MB
        size_t lower = 1024; // 1KB min
        size_t best = 0;
        const int MAX_ITER = 40;
        std::cout << "\n  [Single-allocation probe] binary search to find largest single contiguous cudaMalloc\n";
        for (int iter = 0; iter < MAX_ITER && lower <= upper; ++iter) {
            size_t mid = lower + (upper - lower) / 2;
            void* p = nullptr;
            double tsec = 0.0;
            cudaError_t err = 0;
            bool ok = tryCudaMallocTimed(runt, &p, mid, tsec, err);
            if (ok) {
                // success: record and try higher
                best = mid;
                // free and raise lower bound
                runt.cudaFreeFunc(p);
                lower = mid + 1;
            }
            else {
                // fail: lower the upper bound
                if (err != 0) {
                    //printf("    probe mid=%llu failed (%s)\n", (unsigned long long)mid, runt.getErrorString(err));
                }
                if (mid <= 1024) break;
                upper = (mid - 1);
            }
        }
        if (best == 0) {
            std::cout << "    Could not allocate a reasonably large contiguous block. Try running when GPU is idle.\n";
        }
        else {
            std::cout << "    Largest single contiguous allocation found (approx): " << best << " bytes (" << (best / (1024ULL * 1024ULL)) << " MB)\n";
        }

        // 2) Chunk-allocation until OOM: allocate many chunks of chunkSize to estimate usable total memory
        std::cout << "\n  [Chunk allocation] try allocating many smaller chunks to approach available memory (safe chopping)\n";
        size_t chunkSize = 4 * 1024 * 1024; // start with 4MB chunks
        // reduce chunk if chunk size > best single
        if (best > 0 && chunkSize > best) chunkSize = best;
        if (chunkSize < 1024) chunkSize = 1024;

        std::vector<void*> allocations;
        size_t totalAllocated = 0;
        bool chunkOOM = false;
        for (;;) {
            void* p = nullptr;
            cudaError_t err = runt.cudaMallocFunc(&p, chunkSize);
            if (err == 0 && p != nullptr) {
                allocations.push_back(p);
                totalAllocated += chunkSize;
                // safety: don't try to allocate more than totalMem
                if (totalAllocated + chunkSize > prop.totalGlobalMem) break;
            }
            else {
                // try reducing chunk size to probe remaining space if allocations already happened
                if (allocations.empty()) {
                    // first allocation failed; try smaller chunk (halve) until very small
                    if (chunkSize <= 4096) { chunkOOM = true; break; }
                    chunkSize /= 2;
                    continue;
                }
                else {
                    // stop when next chunk fails
                    chunkOOM = true;
                    break;
                }
            }
        }
        std::cout << "    Total allocated in chunks: " << totalAllocated << " bytes (" << (totalAllocated / (1024ULL * 1024ULL)) << " MB) with chunkSize=" << chunkSize << "\n";

        // free chunked allocations
        for (void* p : allocations) runt.cudaFreeFunc(p);
        allocations.clear();

        // 3) Measure allocation/free time for a typical small allocation (host-timed)
        std::cout << "\n  [Timing] measure host-side cudaMalloc/cudaFree and copy times (pageable vs pinned if available)\n";
        const size_t testCount = 5;
        const size_t smallBytes = 4 * 1024 * 1024; // 4MB
        for (int i = 0; i < testCount; i++) {
            void* p = nullptr;
            double tsec = 0.0;
            cudaError_t err = 0;
            bool ok = tryCudaMallocTimed(runt, &p, smallBytes, tsec, err);
            if (ok) {
                std::cout << "    cudaMalloc(" << smallBytes << ") took " << (tsec * 1000.0) << " ms\n";
                // time cudaFree
                LARGE_INTEGER t0, t1;
                QueryPerformanceCounter(&t0);
                runt.cudaFreeFunc(p);
                QueryPerformanceCounter(&t1);
                double freeSec = qpcToSeconds(t0, t1);
                std::cout << "    cudaFree(" << smallBytes << ") took " << (freeSec * 1000.0) << " ms\n";
            }
            else {
                std::cout << "    cudaMalloc(" << smallBytes << ") failed: " << runt.getErrorString(err) << "\n";
            }
        }

        // 4) Transfer latency/bandwidth tests (H->D, D->H) with pageable and pinned memory
        const size_t smallTransfer = 4096; // 4KB (latency)
        const size_t bigTransfer = 8 * 1024 * 1024; // 8MB (bandwidth)
        auto doTransferTest = [&](size_t bytes, bool usePinned) {
            std::vector<char> hostPageable;
            void* hostPtr = nullptr;
            if (usePinned && runt.cudaHostAllocFunc) {
                cudaError_t err = runt.cudaHostAllocFunc(&hostPtr, bytes, 0); // default flags
                if (err != 0 || hostPtr == nullptr) {
                    std::cout << "    pinned host alloc failed, falling back to pageable\n";
                    hostPageable.resize(bytes);
                    hostPtr = hostPageable.data();
                }
            }
            else {
                hostPageable.resize(bytes);
                hostPtr = hostPageable.data();
            }

            // create device buffer
            void* devBuf = nullptr;
            rc = runt.cudaMallocFunc(&devBuf, bytes);
            if (rc != 0 || devBuf == nullptr) {
                std::cout << "    device alloc for transfer test failed (" << runt.getErrorString(rc) << ")\n";
                if (usePinned && runt.cudaFreeHostFunc && hostPtr && hostPtr != hostPageable.data()) runt.cudaFreeHostFunc(hostPtr);
                return;
            }

            // fill host memory
            memset(hostPtr, 0xAA, bytes);

            // host timing H->D
            LARGE_INTEGER s0, s1;
            QueryPerformanceCounter(&s0);
            rc = runt.cudaMemcpyFunc(devBuf, hostPtr, bytes, cudaMemcpyHostToDevice);
            QueryPerformanceCounter(&s1);
            double hostH2D_ms = qpcToSeconds(s0, s1) * 1000.0;

            // host timing D->H
            QueryPerformanceCounter(&s0);
            rc = runt.cudaMemcpyFunc(hostPtr, devBuf, bytes, cudaMemcpyDeviceToHost);
            QueryPerformanceCounter(&s1);
            double hostD2H_ms = qpcToSeconds(s0, s1) * 1000.0;

            // device-side timing via events if available (more accurate for larger transfers)
            double devElapsedMs = -1.0;
            if (runt.cudaEventCreateFunc && runt.cudaEventRecordFunc && runt.cudaEventElapsedTimeFunc && runt.cudaEventDestroyFunc) {
                void* start = nullptr;
                void* stop = nullptr;
                if (runt.cudaEventCreateFunc(&start) == 0 && runt.cudaEventCreateFunc(&stop) == 0) {
                    // record start, memcpy async is not bound here (we used synchronous cudaMemcpy), but event timing is still useful for gpu ops like memset
                    runt.cudaEventRecordFunc(start, 0);
                    // run a device memcpy (device->device) to measure device-side transfer if enough memory
                    if (runt.cudaMemcpyFunc(devBuf, devBuf, bytes, cudaMemcpyDeviceToDevice) == 0) {
                        runt.cudaEventRecordFunc(stop, 0);
                        runt.cudaEventSynchronizeFunc(stop);
                        float ms = 0.0f;
                        if (runt.cudaEventElapsedTimeFunc(&ms, start, stop) == 0) devElapsedMs = ms;
                    }
                    runt.cudaEventDestroyFunc(start);
                    runt.cudaEventDestroyFunc(stop);
                }
            }

            std::cout << "    Transfer " << bytes << " bytes (" << (usePinned ? "pinned" : "pageable") << "): H->D = " << hostH2D_ms << " ms, D->H = " << hostD2H_ms << " ms";
            if (devElapsedMs >= 0.0) std::cout << ", device-side D2D measured " << devElapsedMs << " ms";
            std::cout << "\n";

            // cleanup
            runt.cudaFreeFunc(devBuf);
            if (usePinned && runt.cudaFreeHostFunc) runt.cudaFreeHostFunc(hostPtr);
        };

        std::cout << "\n  Transfer latency tests (small/big) using pageable and pinned (if available):\n";
        std::cout << "    small transfer (4KB):\n";
        doTransferTest(smallTransfer, false);
        if (runt.cudaHostAllocFunc) doTransferTest(smallTransfer, true);

        std::cout << "    big transfer (8MB):\n";
        doTransferTest(bigTransfer, false);
        if (runt.cudaHostAllocFunc) doTransferTest(bigTransfer, true);

        // 5) Device memory access time (approx) measured using cudaMemset and events
        std::cout << "\n  [Device memory touch] measure cudaMemset on a sizeable buffer to approximate GPU memory access bandwidth\n";
        size_t touchBytes = std::min<size_t>(prop.totalGlobalMem / 4, 32ULL * 1024ULL * 1024ULL); // up to 32MB or 1/4 of memory
        if (touchBytes < 4096) touchBytes = 4096;
        void* dTouch = nullptr;
        rc = runt.cudaMallocFunc(&dTouch, touchBytes);
        if (rc == 0 && dTouch != nullptr) {
            bool usedEventTiming = false;
            double hostMs = 0.0;
            // prefer event timing
            if (runt.cudaEventCreateFunc && runt.cudaEventRecordFunc && runt.cudaEventElapsedTimeFunc && runt.cudaEventSynchronizeFunc) {
                void* start = nullptr; void* stop = nullptr;
                if (runt.cudaEventCreateFunc(&start) == 0 && runt.cudaEventCreateFunc(&stop) == 0) {
                    runt.cudaEventRecordFunc(start, 0);
                    // do memset which runs on GPU
                    runt.cudaMemsetFunc(dTouch, 0x42, touchBytes);
                    runt.cudaEventRecordFunc(stop, 0);
                    runt.cudaEventSynchronizeFunc(stop);
                    float ms = 0;
                    if (runt.cudaEventElapsedTimeFunc(&ms, start, stop) == 0) {
                        usedEventTiming = true;
                        std::cout << "    cudaMemset(" << touchBytes << ") device-time = " << ms << " ms (approx)\n";
                        double bw = double(touchBytes) / (ms / 1000.0) / (1024.0 * 1024.0);
                        std::cout << "    approx device bandwidth during memset = " << bw << " MB/s\n";
                    }
                    runt.cudaEventDestroyFunc(start);
                    runt.cudaEventDestroyFunc(stop);
                }
            }
            if (!usedEventTiming) {
                LARGE_INTEGER t0, t1;
                QueryPerformanceCounter(&t0);
                runt.cudaMemsetFunc(dTouch, 0x42, touchBytes);
                // synchronize to ensure operation complete
                if (runt.cudaDeviceSynchronizeFunc) runt.cudaDeviceSynchronizeFunc();
                QueryPerformanceCounter(&t1);
                double ms = qpcToSeconds(t0, t1) * 1000.0;
                std::cout << "    cudaMemset(" << touchBytes << ") host-time = " << ms << " ms\n";
            }
            runt.cudaFreeFunc(dTouch);
        }
        else {
            std::cout << "    cudaMalloc for touch test failed (" << runt.getErrorString(rc) << ")\n";
        }

        std::cout << "\n-------------------------------------------------------------\n\n";
    }

    runt.unload();
    return 0;
}
#endif



/* to jest pierwsze demo - dziala */
#if 1
// cuda_probe.cpp
// Dynamically load cudart and query devices, compute cores, run small allocate/copy test.
// Works with older cudart on Windows (e.g. Fermi GT540M + older CUDA runtimes).
#define NOMINMAX



#include <windows.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <string>
#include <limits>

typedef int cudaError_t; // runtime returns int-like codes (cudaSuccess == 0)

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int tccDriver;
};

// nvcuda.dll stuff
typedef int CUresult;         // driver API result
typedef int CUdevice;         // device handle (ordinal)
typedef int CUdevice_attribute; // attribute ID

// initialize nvcuda
typedef CUresult(__stdcall* cuInit_t)(unsigned int);


// Function pointer typedefs
typedef cudaError_t(__stdcall* cudaMalloc_t)(void**, size_t);
typedef cudaError_t(__stdcall* cudaFree_t)(void*);
typedef cudaError_t(__stdcall* cudaMemcpy_t)(void*, const void*, size_t, cudaMemcpyKind);
typedef const char* (__stdcall* cudaGetErrorString_t)(cudaError_t);
typedef cudaError_t(__stdcall* cudaGetDeviceProperties_t)(cudaDeviceProp*, int);
typedef cudaError_t(__stdcall* cudaGetDeviceCount_t)(int*);
typedef cudaError_t(__stdcall* cudaDeviceSynchronize_t)();
typedef CUresult(__stdcall* cuDeviceGetAttribute_t)(int* pi, CUdevice_attribute attrib, CUdevice dev);

struct CudaRuntime {
    HMODULE h = nullptr;
    cudaMalloc_t cudaMallocFunc = nullptr;
    cudaFree_t cudaFreeFunc = nullptr;
    cudaMemcpy_t cudaMemcpyFunc = nullptr;
    cudaGetErrorString_t cudaGetErrorStringFunc = nullptr;
    cudaGetDeviceProperties_t cudaGetDevicePropertiesFunc = nullptr;
    cudaGetDeviceCount_t cudaGetDeviceCountFunc = nullptr;
    cudaDeviceSynchronize_t cudaDeviceSynchronizeFunc = nullptr;
    cuDeviceGetAttribute_t cudaDeviceGetAttributeFunc = nullptr;
    cuInit_t cuInitFunc = nullptr;

    HMODULE ndriver = nullptr;

    bool loaded() const { return h != nullptr; }

    bool load() {
        // Try several common cudart dll names (64-bit / 32-bit / versioned names)
        const wchar_t* candidates[] = {
            L"cudart64_80.dll", // older common (CUDA 8)
            L"cudart64_75.dll",
            L"cudart64_72.dll",
            L"cudart64_70.dll",
            L"cudart64_60.dll",
            L"cudart64_50.dll",
            L"cudart64_40_0.dll",
            L"cudart.dll" // fallback: cudart.dll in PATH
        };

        /*
        for (auto& name : candidates) {
            LPCWSTR _name = "C:\\CUDA\\bin\\" + (LPCWSTR)name;
            HMODULE m = LoadLibraryW(_name);
            if (m) {
                h = m;
                break;
            }
        }
        if (!h) return false;
        */

        // Load cudart.dll
        h = LoadLibraryW(L"C:\\CUDA\\bin\\cudart.dll");
        if (!h) {
            std::cerr << "Failed to load cudart.dll!" << GetLastError() << std::endl;
            return false;
        }

        ndriver = LoadLibraryW(L"C:\\Windows\\System32\\nvcuda.dll");
        if (!ndriver)
        {
            std::cerr << " failed to load nvcuda.dll" << GetLastError() << std::endl;
            return false;
        }


        // nvcuda initialization - fixes for ERROR CODE 3
        auto myCuInit = (cuInit_t)GetProcAddress(ndriver, "cuInit");
        if (!myCuInit) {
            std::cerr << "Failed to load cuInit\n";
            return 1;
        }

        CUresult initRes = myCuInit(0);
        if (initRes != 0) {
            std::cerr << "cuInit failed with code " << initRes << "\n";
            return 1;
        }

        // nvcuda
        cudaDeviceGetAttributeFunc = (cuDeviceGetAttribute_t)GetProcAddress(ndriver, "cuDeviceGetAttribute");
        if (!cudaDeviceGetAttributeFunc)
        {
            std::cerr << "Failed to load cuDeviceGetAttribute\n";
            return 1;
        }
        
        int smCount = 0;
        CUdevice dev = 0; // first GPU
        CUdevice_attribute attr = 16; // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT (value = 16 in headers)

        CUresult rc = cudaDeviceGetAttributeFunc(&smCount, attr, dev);
        if (rc == 0) {
            std::cout << "Multiprocessor count = " << smCount << "\n";
        }
        else {
            std::cerr << "cuDeviceGetAttribute failed with code " << rc << "\n";
        }


        // load function pointers (use ANSI names)
        cudaMallocFunc = (cudaMalloc_t)GetProcAddress(h, "cudaMalloc");
        cudaFreeFunc = (cudaFree_t)GetProcAddress(h, "cudaFree");
        cudaMemcpyFunc = (cudaMemcpy_t)GetProcAddress(h, "cudaMemcpy");
        cudaGetErrorStringFunc = (cudaGetErrorString_t)GetProcAddress(h, "cudaGetErrorString");
        cudaGetDevicePropertiesFunc = (cudaGetDeviceProperties_t)GetProcAddress(h, "cudaGetDeviceProperties");
        cudaGetDeviceCountFunc = (cudaGetDeviceCount_t)GetProcAddress(h, "cudaGetDeviceCount");
        cudaDeviceSynchronizeFunc = (cudaDeviceSynchronize_t)GetProcAddress(h, "cudaDeviceSynchronize");

        // Some older runtimes don't have cudaDeviceSynchronize exported (rare), but other APIs are essential.
        if (!cudaMallocFunc || !cudaFreeFunc || !cudaMemcpyFunc || !cudaGetErrorStringFunc || !cudaGetDeviceCountFunc || !cudaGetDevicePropertiesFunc) {
            FreeLibrary(h);
            h = nullptr;
            return false;
        }
        return true;
    }

    void unload() {
        if (h) {
            FreeLibrary(h);
            h = nullptr;
        }
    }

    const char* getErrorString(cudaError_t e) const {
        if (cudaGetErrorStringFunc) return cudaGetErrorStringFunc(e);
        return "Unknown (cudaGetErrorString not available)";
    }
};

// Map compute capability to cores per SM (common mapping used in many references).
// Fermi: CC 2.0 -> 32, CC 2.1 -> 48
// Kepler: 3.x -> 192
// Maxwell: 5.x -> 128
// Pascal: 6.x -> 64 (typical)
// Volta/Turing/Ampere: 7.x/8.x -> 64 (approx, architecture differences exist).
static int coresPerSM(int major, int minor) {
    if (major == 1) {
        return 8; // legacy (1.x)
    }
    if (major == 2) {
        if (minor == 0) return 32;
        return 48; // 2.1 and other 2.x revisions typically 48 for 2.1
    }
    if (major == 3) return 192; // Kepler
    if (major == 5) return 128; // Maxwell
    if (major == 6) return 64;  // Pascal
    if (major >= 7) return 64;  // Volta, Turing, Ampere - use 64 as common baseline (approx).
    return 64; // fallback conservative
}

int main() {
    CudaRuntime runt;
    if (!runt.load()) {
        std::cerr << "Failed to load cudart DLL or required symbols. Make sure a compatible cudart DLL is on PATH.\n";
        return 1;
    }

    // get device count
    int deviceCount = 0;
    cudaError_t rc = runt.cudaGetDeviceCountFunc(&deviceCount);
    if (rc != 0) {
        std::cerr << "cudaGetDeviceCount failed: " << runt.getErrorString(rc) << " (" << rc << ")\n";
        runt.unload();
        return 2;
    }

    if (deviceCount <= 0) {
        std::cerr << "No CUDA devices found.\n";
        runt.unload();
        return 3;
    }

    std::cout << "CUDA device count: " << deviceCount << "\n\n";

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop{};
        rc = runt.cudaGetDevicePropertiesFunc(&prop, dev);
        if (rc != 0) {
            std::cerr << "Failed to get properties for device " << dev << ": " << runt.getErrorString(rc) << "\n";
            continue;
        }

        std::string devName(prop.name);
        int major = prop.major;
        int minor = prop.minor;
        int sm = prop.multiProcessorCount;
        
        int cores_sm = coresPerSM(major, minor);
        long long total_cores = static_cast<long long>(sm) * cores_sm;

        std::cout << " Clock freq test " << prop.clockRate << std::endl;

        std::cout << "Device " << dev << ": " << devName << "\n";
        std::cout << "  Compute capability : " << major << "." << minor << "\n";
        std::cout << "  Multiprocessors    : " << sm << "\n";
        std::cout << "  Cores per SM       : " << cores_sm << "\n";
        std::cout << "  Total CUDA cores   : " << total_cores << "\n";
        std::cout << "  Total Global Mem   : " << prop.totalGlobalMem << " bytes (" << (prop.totalGlobalMem / (1024ULL * 1024ULL)) << " MB)\n";
        std::cout << "  Shared mem / block : " << prop.sharedMemPerBlock << " bytes\n";
        std::cout << "  Warp size          : " << prop.warpSize << "\n";
        std::cout << "  Max threads / block: " << prop.maxThreadsPerBlock << "\n";

        // small runtime test: allocate an int array sized by total_cores (safely capped)
        size_t wantCount = static_cast<size_t>(total_cores);
        if (wantCount == 0) wantCount = 1;
        // cap allocation to something reasonable: at most 1/16th of global memory or 16M ints
        size_t maxIntsByMem = prop.totalGlobalMem / sizeof(int) / 16;
        const size_t MAX_SAFE = 16 * 1024 * 1024; // 16M ints
        size_t allocCount = std::min(wantCount, std::min(maxIntsByMem > 0 ? maxIntsByMem : wantCount, MAX_SAFE));
        if (allocCount == 0) allocCount = 1;

        size_t bytes = allocCount * sizeof(int);
        std::vector<int> hostSrc(allocCount);
        for (size_t i = 0; i < allocCount; ++i) hostSrc[i] = static_cast<int>(i);

        void* devPtr = nullptr;
        rc = runt.cudaMallocFunc(&devPtr, bytes);
        if (rc != 0 || devPtr == nullptr) {
            std::cerr << "  cudaMalloc(" << bytes << ") failed: " << runt.getErrorString(rc) << " (" << rc << ")\n";
            std::cout << "  Skipping copy test for device " << dev << "\n\n";
            continue;
        }

        rc = runt.cudaMemcpyFunc(devPtr, hostSrc.data(), bytes, cudaMemcpyHostToDevice);
        if (rc != 0) {
            std::cerr << "  cudaMemcpy Host->Device failed: " << runt.getErrorString(rc) << " (" << rc << ")\n";
            runt.cudaFreeFunc(devPtr);
            continue;
        }

        // Optional: attempt cudaDeviceSynchronize if available
        if (runt.cudaDeviceSynchronizeFunc) {
            runt.cudaDeviceSynchronizeFunc();
        }

        std::vector<int> hostDst(allocCount, -1);
        rc = runt.cudaMemcpyFunc(hostDst.data(), devPtr, bytes, cudaMemcpyDeviceToHost);
        if (rc != 0) {
            std::cerr << "  cudaMemcpy Device->Host failed: " << runt.getErrorString(rc) << " (" << rc << ")\n";
            runt.cudaFreeFunc(devPtr);
            continue;
        }

        // verify
        bool ok = true;
        for (size_t i = 0; i < allocCount; ++i) {
            if (hostDst[i] != hostSrc[i]) { ok = false; break; }
        }

        if (ok) {
            std::cout << "  Copy test PASSED (ints copied: " << allocCount << ")\n";
        }
        else {
            std::cout << "  Copy test FAILED (data mismatch)\n";
        }

        rc = runt.cudaFreeFunc(devPtr);
        if (rc != 0) {
            std::cerr << "  cudaFree failed: " << runt.getErrorString(rc) << " (" << rc << ")\n";
        }

        std::cout << "\n";
    }

    runt.unload();
    return 0;
}
#endif
