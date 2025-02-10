# SSE and AVX SIMD Vectorization in C

This documentation refers to the source codes used in the folders prefixed with E_, which mainly examine the use of SIMD intrinsic functions in C codes. These intrinsic functions are used when highly optimised code is required and the compiler is unable to automatically vectorize; this is a common problem in complex engineering analysis.

Return to the [main repository documentation](./README.md).


**Table of Contents / Quick Links**

[E1 - Vector Addition using SSE](#sse_vector_add)

<a id="sse_vector_add"></a>
## E1 - Vector Addition using SSE

Here we examine two novel ideas:
* Using posix_memalign to allocate memory along cache boundaries, and
* Using a SSE intrinsic function to help accelerate a vector addition.

To build and run - navigate to the directory holding this example and type "make", i.e.:

```bash
cd E_1_SSE_Addition/
make && ./main.exe
```

### Expected Output

A snippet of the expected output is shown below:

```bash
Result ==> c[0] = 0.000000
Result ==> c[1] = 3.000000
Result ==> c[2] = 6.000000
Result ==> c[3] = 9.000000
Result ==> c[4] = 12.000000
Result ==> c[5] = 15.000000
Result ==> c[6] = 18.000000
Result ==> c[7] = 21.000000
```