<h2>SIMD demo</h2>
<br />
If in <b>main.cpp</b> in line 12 

``` //#define SECOND_TEST_TODO ```
is commented do first test, when is define (uncommented) do second test <br /> Intel i3 on my laptop does not support AVX512. Only 256. <br /><br />

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
