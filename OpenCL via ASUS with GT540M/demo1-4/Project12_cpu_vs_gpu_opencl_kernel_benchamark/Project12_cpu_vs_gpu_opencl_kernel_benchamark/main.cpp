#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>  // For QueryPerformanceCounter and CreateThread

#define DATA_SIZE 100000000  // 100 million
#define NUM_THREADS 4        // Number of threads

// Struct to hold thread arguments
typedef struct {
    int thread_id;   // Thread ID
    float* data;     // Pointer to the data array
    int start;       // Starting index for this thread
    int end;         // Ending index for this thread
} ThreadData;

// Thread function
DWORD WINAPI computeThread(LPVOID lpParam) {
    ThreadData* threadData = (ThreadData*)lpParam;

    for (int idx = threadData->start; idx < threadData->end; idx++) {
        float value = (float)idx;
        // Perform the computation
        threadData->data[idx] = sin(value) + cos(value) + exp(value);
    }

    return 0;
}

int main() {
    // Allocate data array to store the results
    float* data = (float*)malloc(DATA_SIZE * sizeof(float));
    HANDLE threads[NUM_THREADS];
    ThreadData threadData[NUM_THREADS];

    // Variables to store the performance counter values
    LARGE_INTEGER frequency;  // To store ticks per second
    LARGE_INTEGER start, end; // To store the start and end ticks
    double elapsedTime;       // To store the elapsed time

    // Get the frequency of the performance counter (ticks per second)
    QueryPerformanceFrequency(&frequency);

    // Calculate chunk size for each thread
    int chunkSize = DATA_SIZE / NUM_THREADS;

    // Start measuring time
    QueryPerformanceCounter(&start);

    // Create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        threadData[i].thread_id = i;
        threadData[i].data = data;
        threadData[i].start = i * chunkSize;
        // Ensure the last thread processes the remaining elements
        threadData[i].end = (i == NUM_THREADS - 1) ? DATA_SIZE : (i + 1) * chunkSize;

        threads[i] = CreateThread(NULL, 0, computeThread, &threadData[i], 0, NULL);
    }

    // Wait for all threads to finish
    WaitForMultipleObjects(NUM_THREADS, threads, TRUE, INFINITE);

    // End measuring time
    QueryPerformanceCounter(&end);

    // Calculate the elapsed time in seconds
    elapsedTime = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;

    // Print the elapsed time
    printf("\nElapsed Time: %f seconds with %d threads\n", elapsedTime, NUM_THREADS);

    // Clean up: free allocated memory and close handles
    free(data);
    for (int i = 0; i < NUM_THREADS; i++) {
        CloseHandle(threads[i]);
    }

    return 0;
}



#if 0
#include <stdio.h>
#include <math.h>
#include <windows.h>  // For QueryPerformanceFrequency and QueryPerformanceCounter

// Increase the data size to increase the computational workload
#define DATA_SIZE 100000000  // 100 million

int main() {
    // Allocate data array to store the results
    float* data = (float*)malloc(DATA_SIZE * sizeof(float));

    // Variables to store the performance counter values
    LARGE_INTEGER frequency;    // To store ticks per second
    LARGE_INTEGER start, end;   // To store the start and end ticks
    double elapsedTime;         // To store the elapsed time

    // Get the frequency of the performance counter (ticks per second)
    QueryPerformanceFrequency(&frequency);

    // Start measuring time
    QueryPerformanceCounter(&start);

    // Perform the same operations as in the OpenCL kernel
    for (int idx = 0; idx < DATA_SIZE; idx++) {
        // Simulate the operations done in the kernel
        float value = (float)idx;
        float result = sin(value) + cos(value) + exp(value);

        // Store the result in the data array
        data[idx] = result;
    }

    // End measuring time
    QueryPerformanceCounter(&end);

    // Calculate the elapsed time in seconds
    elapsedTime = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;

    // Print the elapsed time
    printf("\nElapsed Time: %f seconds\n", elapsedTime);

    // Free allocated memory
    free(data);

    return 0;
}
#endif



#if 0
#include <stdio.h>
#include <math.h>
#include <windows.h>  // For QueryPerformanceFrequency and QueryPerformanceCounter

#define DATA_SIZE 1024

int main() {
    // Allocate data array to store the results
    float data[DATA_SIZE];

    // Variables to store the performance counter values
    LARGE_INTEGER frequency;    // To store ticks per second
    LARGE_INTEGER start, end;   // To store the start and end ticks
    double elapsedTime;         // To store the elapsed time

    // Get the frequency of the performance counter (ticks per second)
    QueryPerformanceFrequency(&frequency);

    // Start measuring time
    QueryPerformanceCounter(&start);

    // Perform the same operations as in the OpenCL kernel
    for (int idx = 0; idx < DATA_SIZE; idx++) {
        // Simulate the operations done in the kernel
        float value = (float)idx;
        float result = sin(value) + cos(value) + exp(value);

        // Store the result in the data array
        data[idx] = result;
    }

    // End measuring time
    QueryPerformanceCounter(&end);

    // Calculate the elapsed time in seconds
    elapsedTime = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;

    // Print the results
    //for (int i = 0; i < DATA_SIZE; i++) {
   //     printf("Index %d: Result = %f\n", i, data[i]);
    //}

    // Print the elapsed time
    printf("\nElapsed Time: %f seconds\n", elapsedTime);

    return 0;
}
#endif


#if 0
#include <stdio.h>
#include <cmath>

#define DATA_SIZE 1024

int main() {
    // Allocate data array to store the results
    float data[DATA_SIZE];

    // Perform the same operations as in the OpenCL kernel
    for (int idx = 0; idx < DATA_SIZE; idx++) {
        // Simulate the operations done in the kernel
        float value = (float)idx;
        float result = sin(value) + cos(value) + exp(value);

        // Store the result in the data array
        data[idx] = result;
    }

    // Print the results
    for (int i = 0; i < DATA_SIZE; i++) {
        printf("Index %d: Result = %f\n", i, data[i]);
        if (i > 100)
            break;
    }

    return 0;
}
#endif





#if 0 
#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <mutex>

// Define DATA_SIZE and number of iterations
#define DATA_SIZE 1024
#define NUM_THREADS 4

// Global mutex for console output synchronization
std::mutex mtx;

// Function to perform the complex computation for each thread
void compute_step(float* A, float* B, float* C, float* D, float* E, float* F, unsigned int start, unsigned int end, int iterations) {
    for (unsigned int id = start; id < end; id++) {
        float accum = 0.0f;
        for (int i = 0; i < iterations; i++) {
            float a = A[id];
            float b = B[id];
            float c = C[id];
            float d = D[id];
            float e = E[id];
            float f = F[id];

            

            float result1 = (a * b + sin(a + i)) / (cos(b) + 1.0f + i * 0.0001f);
            float result2 = sqrt(fabs(a - b + i * 0.001f)) * exp(-a * b * i);

            accum += (result1 + result2);
            C[id] = result1 + result2 + c;
            D[id] = (result1 * result2) / (result1 + result2 + 0.001f + i * 0.0001f);
            E[id] = pow(result1, 3) + pow(result2, 2) + e;
            F[id] = (C[id] + D[id] + E[id]) / 3.0f + f;

            std::cout << F[id] << std::endl;

            // Accumulate and reset every 100 iterations
            if (i % 100 == 0) {
                F[id] += accum;
                accum = 0.0f;
            }
        }
        F[id] += accum;  // Final addition outside the loop
    }
}

// Function to run the benchmark with multithreading
void run_benchmark_multithreaded(int num_threads, unsigned int n, std::vector<float>& A, std::vector<float>& B, std::vector<float>& C, std::vector<float>& D, std::vector<float>& E, std::vector<float>& F) {
    unsigned int work_per_thread = n / num_threads;
    int total_iterations = 100;  // Only 1 iteration as per the GPU code

    std::vector<std::thread> threads;

    // Create threads and start the computation
    for (int t = 0; t < num_threads; t++) {
        unsigned int start = t * work_per_thread;
        unsigned int end = (t == num_threads - 1) ? n : start + work_per_thread;

        threads.emplace_back(compute_step, A.data(), B.data(), C.data(), D.data(), E.data(), F.data(), start, end, total_iterations);
    }

    // Join all threads to ensure the computation completes
    for (auto& t : threads) {
        t.join();
    }

    // Print final results, limited to first 100 elements
    std::lock_guard<std::mutex> lock(mtx);  // Synchronize console output
    std::cout << "Final results:\n";
    for (unsigned int i = 0; i < n; i++) {
        if (i >= 100) break;
        std::cout << F[i] << " ";
    }
    std::cout << "\n";
}

int main() {
    unsigned int n = DATA_SIZE;  // Set data size to 1024

    // Initialize input vectors with sample data
    std::vector<float> A(n, 1.0f), B(n, 2.0f), C(n, 3.0f), D(n, 4.0f), E(n, 5.0f), F(n, 6.0f);

    // Run the benchmark with 4 threads
    run_benchmark_multithreaded(NUM_THREADS, n, A, B, C, D, E, F);

    return 0;
}

#endif







#if 0

#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>

// Define DATA_SIZE
#define DATA_SIZE 1024

// Simulate the complex calculation performed in the OpenCL kernel
void compute_step(float* A, float* B, float* C, float* D, float* E, float* F, unsigned int start, unsigned int end, int iteration) {
    for (unsigned int id = start; id < end; id++) {
        float a = A[id];
        float b = B[id];
        float c = C[id];
        float d = D[id];
        float e = E[id];
        float f = F[id];

        float result1 = (a * b + sin(a + iteration)) / (cos(b) + 1.0f + iteration * 0.0001f);
        float result2 = sqrt(fabs(a - b + iteration * 0.001f)) * exp(-a * b * iteration);

        C[id] = result1 + result2 + c;
        D[id] = (result1 * result2) / (result1 + result2 + 0.001f + iteration * 0.0001f);
        E[id] = pow(result1, 3) + pow(result2, 2) + e;
        F[id] = (C[id] + D[id] + E[id]) / 3.0f + f;
    }
}

// Function to run computation in parallel with 4 threads and checkpointing
void run_benchmark_with_checkpoints(int num_threads, unsigned int n, std::vector<float>& A, std::vector<float>& B, std::vector<float>& C, std::vector<float>& D, std::vector<float>& E, std::vector<float>& F) {
    int total_iterations = 10000;
    int iteration_checkpoint = 100;
    int iteration_step = total_iterations / iteration_checkpoint;
    unsigned int work_per_thread = n / num_threads;

    for (int step = 0; step < iteration_step; step++) {
        std::vector<std::thread> threads;

        // Run computation on each thread
        for (int t = 0; t < num_threads; t++) {
            unsigned int start = t * work_per_thread;
            unsigned int end = (t == num_threads - 1) ? n : start + work_per_thread;

            threads.emplace_back(compute_step, A.data(), B.data(), C.data(), D.data(), E.data(), F.data(), start, end, step * iteration_checkpoint);
        }

        // Join threads
        for (auto& t : threads) {
            t.join();
        }

        // Simulate checkpoint - reading the results
        std::cout << "Results after step " << step * iteration_checkpoint << ":\n";
        // Optionally, print results for verification
        for (unsigned int i = 0; i < n; i++) {
            if (i < 10)  // Only print first 10 items for brevity
                std::cout << F[i] << " ";
        }
        std::cout << "\n";
    }

    // Final computation after all iterations
    std::cout << "Final results:\n";
    for (unsigned int i = 0; i < n; i++) {
        if (i > 100)
            break;
        std::cout << F[i] << " ";
    }
    std::cout << "\n";
}

int main() {
    unsigned int n = DATA_SIZE;  // Now n = 1024
    std::vector<float> A(n, 1.0f), B(n, 1.0f), C(n, 1.0f), D(n, 1.0f), E(n, 1.0f), F(n, 1.0f);

    // Run the benchmark with 4 threads and checkpoints
    run_benchmark_with_checkpoints(4, n, A, B, C, D, E, F);

    return 0;
}

#endif



#if 0

#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <mutex>
#include <chrono>


// Simulate the complex calculation performed in the OpenCL kernel
void compute_step(float* A, float* B, float* C, float* D, float* E, float* F, unsigned int n, unsigned int start, unsigned int end, int iteration) {
    for (unsigned int id = start; id < end; id++) {
        if (id < n) {
            float a = A[id];
            float b = B[id];
            float c = C[id];
            float d = D[id];
            float e = E[id];
            float f = F[id];

            float result1 = (a * b + sin(a + iteration)) / (cos(b) + 1.0f + iteration * 0.0001f);
            float result2 = sqrt(fabs(a - b + iteration * 0.001f)) * exp(-a * b * iteration);

            C[id] = result1 + result2 + c;
            D[id] = (result1 * result2) / (result1 + result2 + 0.001f + iteration * 0.0001f);
            E[id] = pow(result1, 3) + pow(result2, 2) + e;
            F[id] = (C[id] + D[id] + E[id]) / 3.0f + f;
        }
    }
}

// Function to run computation in parallel with 4 threads and checkpointing
void run_benchmark_with_checkpoints(int num_threads, unsigned int n, std::vector<float>& A, std::vector<float>& B, std::vector<float>& C, std::vector<float>& D, std::vector<float>& E, std::vector<float>& F) {
    int total_iterations = 10000;
    int iteration_checkpoint = 100;
    int iteration_step = total_iterations / iteration_checkpoint;
    unsigned int work_per_thread = n / num_threads;

    for (int step = 0; step < iteration_step; step++) {
        std::vector<std::thread> threads;

        // Run computation on each thread
        for (int t = 0; t < num_threads; t++) {
            unsigned int start = t * work_per_thread;
            unsigned int end = (t == num_threads - 1) ? n : start + work_per_thread;

            threads.emplace_back(compute_step, A.data(), B.data(), C.data(), D.data(), E.data(), F.data(), n, start, end, step * iteration_checkpoint);
        }

        // Join threads
        for (auto& t : threads) {
            t.join();
        }

        // Simulate checkpoint - reading the results
        std::cout << "Results after step " << step * iteration_checkpoint << ":\n";
        // Optionally, print results for verification
        for (unsigned int i = 0; i < n; i++) {
            if (i < 10)  // Only print first 10 items for brevity
                std::cout << F[i] << " ";
        }
        std::cout << "\n";
    }

    // Final computation after all iterations
    std::cout << "Final results:\n";
    for (unsigned int i = 0; i < n; i++) {
        if (i > 100)
            break;
        std::cout << F[i] << " ";
    }
    std::cout << "\n";
}

int main() {
    unsigned int n = 1000000;  // Size of data
    std::vector<float> A(n, 1.0f), B(n, 1.0f), C(n, 1.0f), D(n, 1.0f), E(n, 1.0f), F(n, 1.0f);

    // Run the benchmark with 4 threads and checkpoints
    run_benchmark_with_checkpoints(4, n, A, B, C, D, E, F);

    return 0;
}

#endif





#if 0

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>

// Function to perform the complex benchmark
void complex_benchmark(float* A, float* B, float* C, float* D, float* E, float* F, unsigned int n, unsigned int start, unsigned int end) {
    int iterations = 10000;
    for (unsigned int id = start; id < end; id++) {
        float accum = 0.0f;
        for (int i = 0; i < iterations; i++) {
            if (id < n) {
                float a = A[id];
                float b = B[id];
                float c = C[id];
                float d = D[id];
                float e = E[id];
                float f = F[id];

                float result1 = (a * b + sin(a + i)) / (cos(b) + 1.0f + i * 0.0001f);
                float result2 = sqrt(fabs(a - b + i * 0.001f)) * exp(-a * b * i);

                accum += (result1 + result2);

                C[id] = result1 + result2 + c;
                D[id] = (result1 * result2) / (result1 + result2 + 0.001f + i * 0.0001f);
                E[id] = pow(result1, 3) + pow(result2, 2) + e;
                F[id] = (C[id] + D[id] + E[id]) / 3.0f + f;

                // Checkpoint after every 100 iterations
                if (i % 100 == 0) {
                    F[id] += accum;
                    std::cout << F[id] << " == > " << accum << std::endl;
                    accum = 0.0f;
                   // std::cout << id << " " << i << " " << accum << std::endl;
                    
                }
                
            }
        }
        if (id < n) {
            F[id] += accum;
        }
        //std::cout << F[id] << std::endl;
    }
}

// Benchmark function for testing with multiple threads
void run_benchmark(int num_threads, unsigned int n, std::vector<float>& A, std::vector<float>& B, std::vector<float>& C, std::vector<float>& D, std::vector<float>& E, std::vector<float>& F) {
    unsigned int work_per_thread = n / num_threads;
    std::vector<std::thread> threads;

    auto start_time = std::chrono::high_resolution_clock::now();
    //std::cout << " test-1" << std::endl;

    // Launch threads
    for (int t = 0; t < num_threads; t++) {
        unsigned int start = t * work_per_thread;
        unsigned int end = (t == num_threads - 1) ? n : start + work_per_thread;

        threads.emplace_back(complex_benchmark, A.data(), B.data(), C.data(), D.data(), E.data(), F.data(), n, start, end);
        std::cout << "                         ========== test " << F.data() << std::endl;
    }

    // Join threads
    for (auto& t : threads) {
        t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Execution time with " << num_threads << " threads: " << elapsed.count() << " seconds.\n";
}

int main() {
    //unsigned int n = 1000000;  // Adjust size as needed
    unsigned int n = 1000;
    std::vector<float> A(n, 1.0f), B(n, 1.0f), C(n, 1.0f), D(n, 1.0f), E(n, 1.0f), F(n, 1.0f);

    // Run the benchmark with 1, 2, 3, and 4 threads
    for (int num_threads = 1; num_threads <= 4; num_threads++) {
        run_benchmark(num_threads, n, A, B, C, D, E, F);
    }

    //std::cout << F << std::endl;

    return 0;
}

#endif
