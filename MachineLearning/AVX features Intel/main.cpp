#include <intrin.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <Windows.h>
#include <math.h>

#pragma intrinsic(__rdtsc)

//#define SECOND_TEST_TODO

struct MyClass {
    void __vectorcall mymethod();
};
void MyClass::mymethod() { return; }
typedef __m256 (__vectorcall* vcfnptr)(double, double, double, double);

typedef struct {
    __m256 x;
    __m256 y;
    __m256 z;
} hva3;    // 3 element HVA type on __m256

typedef struct {
    __m128 array[2];
} hva2;    // 2 element HVA type on __m128

typedef struct {
    __m256 array[4];
} hva4;    // 4 element HVA type on __m256

// High-precision timer
double measure_time(LARGE_INTEGER start, LARGE_INTEGER end, LARGE_INTEGER freq) {
    return (double)(end.QuadPart - start.QuadPart) * 1000000.0 / freq.QuadPart; // Milliseconds
}

// Print __m128
void print_m128(const char* name, __m128 v) {
    float values[4];
    _mm_storeu_ps(values, v);
    printf("%s = { %f, %f, %f, %f }\n", name, values[0], values[1], values[2], values[3]);
}

// Print __m256
void print_m256(const char* name, __m256 v) {
    float values[8];
    _mm256_storeu_ps(values, v);
    printf("%s = { %f, %f, %f, %f, %f, %f, %f, %f }\n", name,
        values[0], values[1], values[2], values[3],
        values[4], values[5], values[6], values[7]);
}

// Function 1: Matrix-vector multiplication
__m128 __vectorcall example1(__m128 a, __m128 b, __m256 c, __m128 d, __m256 e) {
    // Multiply a 4x4 matrix (in `b` and `d`) by a 4-element vector (`a`)
    __m128 row1 = _mm_mul_ps(b, a);
    __m128 row2 = _mm_mul_ps(d, a);
    return _mm_add_ps(row1, row2);
}

// Function 2: Dot product
__m256 __vectorcall example2(int a, __m128 b, int c, __m128 d, __m256 e, float f, int g) {
    __m128 dot_product = _mm_dp_ps(b, d, 0xFF);  // Compute dot product of b and d
    __m256 scaled = _mm256_mul_ps(e, _mm256_set1_ps(f)); // Scale e by f
    return scaled;
}

// Function 3: Matrix multiplication
__m128 __vectorcall example3(int a, hva2 b, int c, int d, int e) {
    // Multiply two 2x2 matrices stored in `b`
    __m128 result = _mm_add_ps(
        _mm_mul_ps(b.array[0], _mm_set1_ps(c)),
        _mm_mul_ps(b.array[1], _mm_set1_ps(d))
    );
    return result;
}

// Function 4: Cross product and scalar scaling
float __vectorcall example4(int a, float b, hva4 c, __m128 d, int e) {
    __m256 cross = _mm256_mul_ps(c.array[0], c.array[1]); // Simulate cross product
    __m256 scaled = _mm256_mul_ps(cross, _mm256_set1_ps(b));
    float result[8];
    _mm256_storeu_ps(result, scaled);
    return result[0]; // Return the first element
}

// Function 5: Vector sum
int __vectorcall example5(int a, hva2 b, int c, hva4 d, int e) {
    __m256 sum = _mm256_add_ps(d.array[0], d.array[1]);
    float result[8];
    _mm256_storeu_ps(result, sum);
    return (int)(result[0] + c + e); // Example sum
}

// Function 6: Large matrix addition
hva4 __vectorcall example6(hva2 a, hva4 b, __m256 c, hva2 d) {
    hva4 result;
    for (int i = 0; i < 4; i++) {
        result.array[i] = _mm256_add_ps(b.array[i], c);
    }
    return result;
}

void scalar_matrix_mult(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}
//
//void simd_matrix_mult(float* A, float* B, float* C, int N) {
//    for (int i = 0; i < N; i++) {
//        for (int j = 0; j < N; j++) {
//            __m256 sum = _mm256_setzero_ps();
//            for (int k = 0; k < N; k += 8) { // Process 8 elements at once
//                __m256 a = _mm256_loadu_ps(&A[i * N + k]);
//                __m256 b = _mm256_loadu_ps(&B[k * N + j]);
//                sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
//            }
//            _mm256_storeu_ps(&C[i * N + j], sum);
//        }
//    }
//}

// VER 1
// Scalar matrix multiplication
//void scalar_matrix_mult(float* A, float* B, float* C, int N) {
//    for (int i = 0; i < N; i++) {
//        for (int j = 0; j < N; j++) {
//            C[i * N + j] = 0;
//            for (int k = 0; k < N; k++) {
//                C[i * N + j] += A[i * N + k] * B[k * N + j];
//            }
//        }
//    }
//}

//void scalar_matrix_mult(float* A, float* B, float* C, int N) {
//    for (int i = 0; i < N; i++) {
//        for (int j = 0; j < N; j++) {
//            float sum = 0.0f;
//            for (int k = 0; k < N; k += 8) { // Process 8 elements at once
//                float partial_sum[8] = { 0.0f };
//                // Load elements manually for 8-wide block
//                for (int p = 0; p < 8 && k + p < N; p++) { // Handle edge cases when N is not divisible by 8
//                    partial_sum[p] = A[i * N + k + p] * B[(k + p) * N + j];
//                }
//                // Sum the partial results to mimic _mm256_add_ps and _mm256_storeu_ps behavior
//                for (int p = 0; p < 8; p++) {
//                    sum += partial_sum[p];
//                }
//            }
//            C[i * N + j] = sum;
//        }
//    }
//}


// SIMD matrix multiplication
void simd_matrix_mult(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            __m256 sum = _mm256_setzero_ps();
            for (int k = 0; k < N; k += 8) { // Process 8 elements at once
                __m256 a = _mm256_loadu_ps(&A[i * N + k]);
                __m256 b = _mm256_loadu_ps(&B[k * N + j]);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
            }
            float temp[8];
            _mm256_storeu_ps(temp, sum);
            C[i * N + j] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
        }
    }
}

// Utility to initialize matrices with random values
void initialize_matrix(float* matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = (float)(rand() % 10 + 1); // Random values between 1 and 10
    }
}

// function to initialize A nad B
void initialize_matrix_AandB(float* matrixA, float* matrixB, int N) {
    for (int i = 0; i < N * N; i++) {
        float val = (float)(rand() % 10 + 1);  // Random values between 1 and 10
        matrixA[i] = val;
        matrixB[i] = val;
    }
}

#define DEBUG 0

// Utility to print matrices
void print_matrix(const char* label, float* matrix, int N) {
#if DEBUG
    printf("%s:\n", label);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6.2f ", matrix[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
#endif
}

float dot_product_scalar(float* a, float* b, int size) {
    float dot = 0.0f;
    for (int i = 0; i < size; i++) {
        dot += a[i] * b[i];  // Multiply corresponding elements and accumulate
    }
    return dot;
}

float *get_arr(const int size, const float val)
{
    const int SIZE = size; // 8 
    float* ret = (float*)malloc(SIZE * sizeof(float)); // Dynamically allocate memory
    if (!ret) {
        printf("Memory allocation failed\n");
        return NULL; // Return NULL if allocation fails
    }

    for (int i = 0; i < SIZE; i++) {
        ret[i] = val; // Initialize with a value - 5.0f
    }

    return ret; // Return pointer to allocated memory
}

void print_bits(int value) {
    printf("Bits of %d (0x%08X):\n", value, value);
    for (int i = 31; i >= 0; i--) { // Start from the most significant bit
        printf("%d", (value >> i) & 1); // Shift and mask to extract the bit
        if (i % 8 == 0) {
            printf(" "); // Add space every 8 bits for readability
        }
    }
    printf("\n");
}

/* test for AVX 256 */
void check_avx_support_256() {
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);

    int avxSupported = (cpuInfo[2] & (1 << 28)) != 0; // Check AVX bit
    int fmaSupported = (cpuInfo[2] & (1 << 12)) != 0; // Check FMA bit

    if (avxSupported) {
        printf("AVX is supported.\n");
    }
    else {
        printf("AVX is not supported.\n");
    }

    if (fmaSupported) {
        printf("FMA is supported.\n");
    }
    else {
        printf("FMA is not supported.\n");
    }
}


#ifndef SECOND_TEST_TODO
// Initialize matrices and vectors
void initialize_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 100) / 10.0f; // Random values between 0.0 and 10.0
    }
}

void initialize_vector(float* vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = (float)(rand() % 100) / 10.0f;
    }
}
#endif

// Scalar implementation of XW + B
void scalar_xw_plus_b(float* X, float* W, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += X[i * N + k] * W[k * K + j];
            }
            C[i * K + j] = sum + B[j];
        }
    }
}

// AVX-256 implementation of XW + B
void avx256_xw_plus_b(float* X, float* W, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {          // Loop over rows of X
        for (int j = 0; j < K; j++) {      // Loop over columns of W
            __m256 sum = _mm256_setzero_ps(); // Initialize sum to zero
            for (int k = 0; k < N; k += 8) {  // Process 8 elements of X and W at a time
                __m256 x_vec = _mm256_loadu_ps(&X[i * N + k]);       // Load 8 elements of X
                __m256 w_vec = _mm256_loadu_ps(&W[k * K + j]);       // Load 8 elements of W
                //sum = _mm256_fmadd_ps(x_vec, w_vec, sum);            // Perform fused multiply-add
                sum = _mm256_add_ps(sum, _mm256_mul_ps(x_vec, w_vec)); // Separate multiply and add

            }
            // Horizontal addition of the 8 elements in sum
            float temp[8];
            _mm256_storeu_ps(temp, sum);
            float dot_product = temp[0] + temp[1] + temp[2] + temp[3] +
                temp[4] + temp[5] + temp[6] + temp[7];
            C[i * K + j] = dot_product + B[j]; // Add bias
        }
    }
}

// Timer function
double measure_time(void (*func)(float*, float*, float*, float*, int, int, int),
    float* X, float* W, float* B, float* C, int M, int N, int K) {
    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);

    func(X, W, B, C, M, N, K);

    QueryPerformanceCounter(&end);
    return (double)(end.QuadPart - start.QuadPart) / freq.QuadPart;
}

/* check AVX 512 support */
void check_avx512_support(const int _eax) {
    int cpuInfo[4];
    __cpuid(cpuInfo, _eax);
    if (cpuInfo[1] & (1 << 5)) {
        printf("AVX-512 is supported.\n");
    }
    else {
        printf("AVX-512 is NOT supported.\n");
    }
    print_bits(cpuInfo[0]);
    print_bits(cpuInfo[1]);
    print_bits(cpuInfo[2]);
    print_bits(cpuInfo[3]);
}

int __cdecl main(void)
{

    // cechk AVX-512 support
    check_avx512_support(1); // eax 1, ecx 0
    check_avx512_support(7); // eax 7, ecx 0
    printf("-----------------------------\n");

    // avx 256
    check_avx_support_256();
     // Dimensions
#ifndef SECOND_TEST_TODO 
#if 1 // default example
    const int M = 512; // Rows of X
    const int N = 256; // Columns of X and rows of W
    const int K = 128; // Columns of W
#endif
#endif

#ifndef SECOND_TEST_TODO 
#if 0
    const int M = 1024; // Rows of X
    const int N = 1024; // Columns of X and rows of W
    const int K = 1024; // Columns of W
#endif 
#endif

#ifndef SECOND_TEST_TODO
    // Allocate memory
    float* X = (float*)_aligned_malloc(M * N * sizeof(float), 32);
    float* W = (float*)_aligned_malloc(N * K * sizeof(float), 32);
    float* B = (float*)_aligned_malloc(K * sizeof(float), 32);
    float* C_scalar = (float*)_aligned_malloc(M * K * sizeof(float), 32);
    float* C_avx256 = (float*)_aligned_malloc(M * K * sizeof(float), 32);

    if (!X || !W || !B || !C_scalar || !C_avx256) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Initialize matrices and vectors
    srand((unsigned int)time(NULL));
    initialize_matrix(X, M, N);
    initialize_matrix(W, N, K);
    initialize_vector(B, K);

    // Measure scalar performance
    double scalar_time = measure_time(scalar_xw_plus_b, X, W, B, C_scalar, M, N, K);

    // Measure AVX-256 performance
    double avx256_time = measure_time(avx256_xw_plus_b, X, W, B, C_avx256, M, N, K);

    // Verify results
    int correct = 1;
    for (int i = 0; i < M * K; i++) {
        if (fabs((float)C_scalar[i] - (float)C_avx256[i]) > 1e-5) {
            correct = 0;
            printf("Mismatch at index %d: scalar = %f, avx256 = %f\n", i, C_scalar[i], C_avx256[i]);
            break;
        }
    }

    // Print results
    printf("Scalar Time: %f seconds\n", scalar_time);
    printf("AVX-256 Time: %f seconds\n", avx256_time);
    printf("Speedup: %.2fx\n", scalar_time / avx256_time);
    printf("Results Match: %s\n", correct ? "Yes" : "No");

    // Free allocated memory
    _aligned_free(X);
    _aligned_free(W);
    _aligned_free(B);
    _aligned_free(C_scalar);
    _aligned_free(C_avx256);


    printf("-----------------------------\n");

    unsigned __int64 start1, end1;
    start1 = __rdtsc();
    printf_s("%I64d ticks\n", start1);

    //printf("%f \n", *get_arr(8, 5.0));

    float* arr = get_arr(8, 5.0);
    if (!arr) return 1; // Exit if allocation failed

    // Print the first element
    printf("%f \n", arr[0]);

    // Access elements using the function pattern: get_arr().get(0)
    for (int i = 0; i < 8; i++) {
        printf("arr[%d] = %f\n", i, arr[i]);
    }

    float dot = dot_product_scalar(arr, arr, 8);

    printf("Dot product (scalar): %.2f\n", dot);

    // Free the allocated memory
    free(arr);

    end1 = __rdtsc();
    printf_s("%I64d %I64d %I64d ticks\n", start1, end1, (end1 - start1));
#endif

    /*----------------------------------*/
    // prevent redefinition N, B etc
    // to test for AVX 256 first
   
#ifdef SECOND_TEST_TODO
    hva4 h4;
    hva2 h2;
    int i;
    float f;
    __m128 a, b, d;
    __m256 c, e;

    LARGE_INTEGER freq, start, end;

    QueryPerformanceFrequency(&freq);

    a = b = d = _mm_set1_ps(3.0f);
    c = e = _mm256_set1_ps(5.0f);
    h2.array[0] = _mm_set1_ps(6.0f);
    h2.array[1] = _mm_set1_ps(8.0f);
    for (int i = 0; i < 4; i++) {
        h4.array[i] = _mm256_set1_ps(7.0f + i);
    }

    QueryPerformanceCounter(&start);
    b = example1(a, b, c, d, e);
    QueryPerformanceCounter(&end);
    print_m128("Result of example1", b);
    printf("Execution time: %.3f ms\n", measure_time(start, end, freq));

    QueryPerformanceCounter(&start);
    e = example2(1, b, 3, d, e, 6.0f, 7);
    QueryPerformanceCounter(&end);
    print_m256("Result of example2", e);
    printf("Execution time: %.3f ms\n", measure_time(start, end, freq));

    QueryPerformanceCounter(&start);
    d = example3(1, h2, 3, 4, 5);
    QueryPerformanceCounter(&end);
    print_m128("Result of example3", d);
    printf("Execution time: %.3f ms\n", measure_time(start, end, freq));

    QueryPerformanceCounter(&start);
    f = example4(1, 2.0f, h4, d, 5);
    QueryPerformanceCounter(&end);
    printf("Result of example4 = %f\n", f);
    printf("Execution time: %.3f ms\n", measure_time(start, end, freq));

    QueryPerformanceCounter(&start);
    i = example5(1, h2, 3, h4, 5);
    QueryPerformanceCounter(&end);
    printf("Result of example5 = %d\n", i);
    printf("Execution time: %.3f ms\n", measure_time(start, end, freq));

    QueryPerformanceCounter(&start);
    h4 = example6(h2, h4, c, h2);
    QueryPerformanceCounter(&end);
    printf("Result of example6:\n");
    for (int i = 0; i < 4; i++) {
        char name[16];
        snprintf(name, sizeof(name), "  array[%d]", i);
        print_m256(name, h4.array[i]);
    }
    printf("Execution time: %.3f ms\n", measure_time(start, end, freq));

    // edge

    const int N = 256; // Matrix size (N x N), adjust for larger tests
    float* A = (float*)_aligned_malloc(N * N * sizeof(float), 32);
    float* B = (float*)_aligned_malloc(N * N * sizeof(float), 32);
    float* C_scalar = (float*)_aligned_malloc(N * N * sizeof(float), 32);
    float* C_simd = (float*)_aligned_malloc(N * N * sizeof(float), 32);

    // Initialize matrices
    srand((unsigned int)time(NULL));
    //initialize_matrix(A, N);
    //initialize_matrix(B, N); // if you want to 2 diffrent matix 
    // but here I want to 2 equal initialization of A na B matrix
    initialize_matrix_AandB(A, B, N);

    // Print matrices (optional for debugging)
    print_matrix("Matrix A", A, N);
    print_matrix("Matrix B", B, N);

    // Measure time for scalar multiplication
    //LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);

    QueryPerformanceCounter(&start);
    scalar_matrix_mult(A, B, C_scalar, N);
    QueryPerformanceCounter(&end);
    double scalar_time = measure_time(start, end, freq);

    // Measure time for SIMD multiplication
    QueryPerformanceCounter(&start);
    simd_matrix_mult(A, B, C_simd, N);
    QueryPerformanceCounter(&end);
    double simd_time = measure_time(start, end, freq);

    // Print results
    print_matrix("Result (Scalar)", C_scalar, N);
    print_matrix("Result (SIMD)", C_simd, N);

    printf("Scalar Time: %.3f ms\n", scalar_time);
    printf("SIMD Time: %.3f ms\n", simd_time);
    printf("Speedup: %.2fx\n", scalar_time / simd_time);

    // Cleanup
    _aligned_free(A);
    _aligned_free(B);
    _aligned_free(C_scalar);
    _aligned_free(C_simd);
#endif

    return 0;
}
