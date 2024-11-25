<h2>SIMD demo</h2>
Without using GPU this might be speed up some matrix calculations. But on my laptop I have install Intel i3 from 2011 rev. So, this is not modern CPU and AVX 512 and other things is not supported. And keep in mind, in this main.cpp in line 310 there is 

```
sum = _mm256_add_ps(sum, _mm256_mul_ps(x_vec, w_vec));
```

This is without FMA support. With FMA this line is 309 and look like that

```
sum = _mm256_fmadd_ps(x_vec, w_vec, sum);
```

<b>But in this demo is many bugs to fix and things "TODO" </b> - these examples 1-6 from uncommented version is to fix. Dot product is wrong etc.

<br /><br />
If in <b>main.cpp</b> in line 12 

``` //#define SECOND_TEST_TODO ```
is commented do first test, when is define (uncommented) do second test <br /> Intel i3 on my laptop does not support AVX512. Only 256. <br /><br />

When commented //#define SECOND_TEST_TODO
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/AVX%20features%20Intel/434%20-%2024-11-2024%20-%20todo%20avx%20256.png?raw=true)

When uncommented #define SECOND_TEST_TODO
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/AVX%20features%20Intel/434%20-%2024-11-2024%20-%20todo%20avx%20ifndef%20second%20test.png?raw=true)

[1] https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/intrinsics-for-arithmetic-operations-002.html<br />
[2] https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/mm256-add-ps.html<br />
[3] https://cdrdv2.intel.com/v1/dl/getContent/767250?fileName=cpp-compiler_developer-guide-reference_2021.8-767249-767250.pdf<br />
[4] https://pytorch.org/docs/stable/generated/torch.dot.html<br />
[5] https://learn.microsoft.com/pl-pl/cpp/intrinsics/rdtsc?view=msvc-170<br />
[6] https://learn.microsoft.com/pl-pl/cpp/intrinsics/cpuid-cpuidex?view=msvc-170<br />
[7] https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/fabs-fabsf-fabsl?view=msvc-170<br />
[8] https://en.wikipedia.org/wiki/CPUID#cite_note-46<br />
[9] https://en.wikipedia.org/wiki/Advanced_Vector_Extensions<br />
[10] https://github.com/fordsfords/rdtsc/blob/main/rdtsc.c<br />
