# OpenMP Code Overview

This document outlines the codes in the repository focusing on the use of OpenMP for parallel computation.

Return to the [main repository documentation](./README.md).

**Table of Contents / Quick Links**

[7 - OpenMP Hello World](#omp_hello_world)  
[8 - OpenMP Improved Hello World](#omp_hello_again)  
[9 - Parallel Vector x Constant multiplication](#omp_vector_constant)  
[10 - The OMP For Pragma](#omp_for)  
[11 - A simple dot product with OMP](#omp_dot_product)  

<a id="omp_hello_world"></a>
## 7 - OpenMP Hello World

Your first OpenMP Hello World code. Let's find out how many threads by default your current environment supports.

To build and run - navigate to the directory holding this example and type "make", i.e.:

```bash
cd 7_OMP_Hello_World
make && ./main.exe
```

### Expected Output

The number of "Hello!" messages you will see depends on your machine. Assuming your CPU supports 4 OpenMP threads natively, you'll see this:

```bash
Hello!
Hello!
Hello!
Hello!
```

<a id="omp_hello_again"></a>
## 8 - OpenMP Improved Hello World

This time, we aim to exercise better control over how many threads we use, and better identify the threads we employ.

To build and run - navigate to the directory holding this example and type "make", i.e.:

```bash
cd 8_OMP_Hello_Again
make && ./main.exe
```

### Expected Output

The number of "Hello!" messages you will see depends on your machine. Assuming your CPU supports 4 OpenMP threads natively, you'll see this:

```bash
Hello again from thread 0
Hello again from thread 1
```
<a id="omp_vector_constant"></a>
## 9 - OpenMP Vector Multiplication by Constant

Finally, a useful computation. Here we take a small array and multiply the elements of the array by a constant through the division of work across multiple threads.

To build and run - navigate to the directory holding this example and type "make", i.e.:

```bash
cd 9_OMP_Vector_Multiply_Constant
make && ./main.exe
```

### Expected Output

```bash
Original value of a[0] = 0
Original value of a[1] = 1
Original value of a[2] = 2
Original value of a[3] = 3
Original value of a[4] = 4
Original value of a[5] = 5
Original value of a[6] = 6
Original value of a[7] = 7
Final value of a[0] = 0
Final value of a[1] = 0.5
Final value of a[2] = 1
Final value of a[3] = 1.5
Final value of a[4] = 2
Final value of a[5] = 2.5
Final value of a[6] = 3
Final value of a[7] = 3.5
```

<a id="omp_for"></a>
## 10 - OpenMP For Pragma

This demo covers the use of the **omp for** pragma, and two scheduling options (dynamc, static).

To build and run - navigate to the directory holding this example and type "make", i.e.:

```bash
cd 10_OMP_For_demo
make && ./main.exe
```

### Expected Output

This can be random, as it depends on your system and hardware scheduling - but your output should take the form:

```bash
Static element thread[0] = 0, Dynamic element thread[0] = 0
Static element thread[1] = 0, Dynamic element thread[1] = 0
Static element thread[2] = 1, Dynamic element thread[2] = 0
Static element thread[3] = 1, Dynamic element thread[3] = 0
Static element thread[4] = 2, Dynamic element thread[4] = 0
Static element thread[5] = 2, Dynamic element thread[5] = 0
Static element thread[6] = 3, Dynamic element thread[6] = 0
Static element thread[7] = 3, Dynamic element thread[7] = 0
Static element thread[8] = 0, Dynamic element thread[8] = 0
Static element thread[9] = 0, Dynamic element thread[9] = 0
Static element thread[10] = 1, Dynamic element thread[10] = 0
Static element thread[11] = 1, Dynamic element thread[11] = 0
```

<a id="omp_dot_product"></a>
## 11 - A simple dot product using OpenMP

Here we use openmp to accelerate the computation of a simple dot product, using
both a standard omp for loop, and an **openmp for reduction**.

To build and run - navigate to the directory holding this example and type "make", i.e.:

```bash
cd 11_OMP_Dot_Product
make && ./main.exe
```

### Expected Output

```bash
Dot product of a,b = 170688
Dot product of a,b using openmp = 170688
```