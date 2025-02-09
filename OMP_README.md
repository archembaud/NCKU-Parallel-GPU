# OpenMP Code Overview

This document outlines the codes in the repository focusing on the use of OpenMP for parallel computation.

Return to the [main repository documentation](./README.md).

**Table of Contents / Quick Links**

[7 - OpenMP Hello World](#omp_hello_world)  
[8 - OpenMP Improved Hello World](#omp_hello_again)  

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
