# Introduction to Multi-Core CPU and GPU Computing
Codes used as part of a course on parallel and GPU computing. These provide a basic introduction to parallel compute with a focus on scientific and mathematical computing. The goal is to provide a means for engineering and science students to write code which executes faster on modern hardware.

These codes can be subdivided into 5 categories:

## Basic C introduction codes

These codes are designed to give a basic overview of the C programming language, how these are compiled and built, and finally their execution in a standard linux environment.

[1 - Hello World](#hello_world)  
[2 - Simple Functions in C](#simple_functions)  
[3 - Arrays in C using the stack](#stack_arrays)  
[4 - Arrays in C allocated on the heap](#heap_arrays)  
[5 - Arrays allocated externally in a function](#arrays_allocation_in_function)  
[6 - Saving to File, Viewing Results](#save_fo_file)  


## Parallel processing using OpenMP

Please refer to the [OpenMP readme file](./OMP_README.md).

## Parallel processing using GPU (CUDA)

Please refer to the [GPU readme file](./GPU_README.md).

## Mathematics and Engineering Applications

Please refer to the [Algorithms readme file](./ALGORITHMS.md).

## SIMD Vectorization using SSE and AVX

Please refer to the [SIMD readme file](./SIMD_README.md).

## Case Studies

There are (at least planned) several case studies, which are larger problems looking at solving a real life problem, or perhaps a problem which might be slightly beyond the scope of the NCKU lecture series.

Each one has its own README:

* 3D Heat Transfer simulation of a heat sink. See [this readme file](./C_1_3D_HT/README.md)  
* 2D Heat Transfer simulation of a bicycle disk brake. See [this readme file.](./C_2_Disk_Brake/README.md)  
* 1D Heat Transfer simluation using 2x GPU cards using CUDA with OpenMP. See [this readme file.](./C_3_Multi-GPU_FTCS/README.md)


## Some General Hints and Tips

* If you don't have your own computer - never fear, you can use the department machines.
Using your own laptop is highly recommended, however.
* If you have your own Windows laptop - install Ubuntu via WSL. The material presented here assumes a linux environment, and WSL is an excellent alternative to a native linux distribution.
* If you have your own Laptop - install VS Code as your editor, and then install the remote ssh extension.
This will allow you to remotely connect to the Department's GPU server and write code in a more familiar environment (for some).
```bash
https://code.visualstudio.com/docs/remote/ssh
```
* You can transfer folders to the remote server using the *scp* commmand - for instance, let's say you want
to transfer the G_2_MemCpy folder to the remote, type:
```bash
 scp -r ./G_2_MemCpy/ YOUR_USER_NAME@140.116.31.227:~
```
This will make a copy of the folder in your home directory. Don't forget to replace YOUR_USER_NAME with your
user name.




<a id="hello_world"></a>
## 1 - Hello World

The first code used as part of this course; it serves to give a demonstration on:
* Using makefiles to i) compile, and then ii) link and build a C code.
* Writing a C code which accepts command line arguments.

Before running the commands below, make sure you are in the correct directory. From the repository root directory, type:

```bash
cd 1_Hello_World
```

To build:
```bash
make
```
To execute:
```bash
./main.exe
```

To remove the executable (main.exe) and built object file (main.o):
```bash
make clean
```
To inspect the disassembled object:

```bash
objdump -d main.o
```

### Expected output

If executed with **"./main.exe test"** the expected output is:

```bash
Hello world program called with 2 arguments
Argument 0 is: ./main.exe
Argument 1 is: test
```

<a id="simple_functions"></a>
## 2 - Simple Functions

The second code used as part of this course; it serves to give a demonstration on:
* Using makefiles to i) compile, and then ii) link and build a slightly more complex code (with multiple source files) by first compiling all sources, and them linking them into a single executable.
* Writing a C code employing a simple function.

Before running the commands below, make sure you are in the correct directory. From the repository root directory, type:

```bash
cd 2_Simple_Function
```

To build:
```bash
make
```
To execute:
```bash
./main.exe
```

To remove the executable (main.exe) and built object file (main.o):
```bash
make clean
```
To inspect the disassembled object:

```bash
objdump -d main.o
```

### Expected output

If executed with **"./main.exe test"** the expected output is:

```bash
The value of 1 + -2 is -1
```

<a id="stack_arrays"></a>
## 3 - Creating arrays on the stack

The third code used as part of this course; it serves to give a demonstration on:
* Using makefiles to i) compile, and then ii) link a simple code, with a new library included (math, -lm)
* Writing a C code which creates a few simple arrays and does some simple computations on them.

Before running the commands below, make sure you are in the correct directory. From the repository root directory, type:

```bash
cd 3_Stack_Arrays
```

To build:
```bash
make
```
To execute:
```bash
./main.exe
```

### Expected output

If executed with **"./main.exe test"** the expected output is:

```bash
Sqrt of (0+0) = 0.000000e+00
Sqrt of (1+2) = 1.732051e+00
Sqrt of (2+4) = 2.449490e+00
Sqrt of (3+6) = 3.000000e+00
Sqrt of (4+8) = 3.464102e+00
Sqrt of (5+10) = 3.872983e+00
Sqrt of (6+12) = 4.242640e+00
Sqrt of (7+14) = 4.582576e+00
Sqrt of (8+16) = 4.898980e+00
Sqrt of (9+18) = 5.196152e+00
Sqrt of (10+20) = 5.477226e+00
Sqrt of (11+22) = 5.744563e+00
Sqrt of (12+24) = 6.000000e+00
Sqrt of (13+26) = 6.244998e+00
Sqrt of (14+28) = 6.480741e+00
Sqrt of (15+30) = 6.708204e+00
Sqrt of (16+32) = 6.928203e+00
Sqrt of (17+34) = 7.141428e+00
Sqrt of (18+36) = 7.348469e+00
Sqrt of (19+38) = 7.549834e+00
Sqrt of (20+40) = 7.745967e+00
Sqrt of (21+42) = 7.937254e+00
Sqrt of (22+44) = 8.124039e+00
Sqrt of (23+46) = 8.306623e+00
Sqrt of (24+48) = 8.485281e+00
Sqrt of (25+50) = 8.660254e+00
Sqrt of (26+52) = 8.831760e+00
Sqrt of (27+54) = 9.000000e+00
Sqrt of (28+56) = 9.165152e+00
Sqrt of (29+58) = 9.327379e+00
```


<a id="heap_arrays"></a>
## 4 - Creating arrays on the heap

The fourth code used as part of this course; it serves to give a demonstration on:
* Writing a C code which creates a few simple arrays using malloc, and
* Demonstrates the use of valgrind for memory leak detection.

Before running the commands below, make sure you are in the correct directory. From the repository root directory, type:

```bash
cd 4_Heap_Arrays
```

To build:
```bash
make
```
To execute:
```bash
./main.exe
```
To check for memory leaks with Valgrind:
```bash
valgrind ./main.exe
```


### Expected output

If executed with **"valgrind ./main.exe"** the expected output is:

```bash
==375108== Memcheck, a memory error detector
==375108== Copyright (C) 2002-2022, and GNU GPLd, by Julian Seward et al.
==375108== Using Valgrind-3.22.0 and LibVEX; rerun with -h for copyright info
==375108== Command: ./main.exe
==375108== 
Sqrt of (0+0) = 0.000000e+00
Sqrt of (1+2) = 1.732051e+00
Sqrt of (2+4) = 2.449490e+00
Sqrt of (3+6) = 3.000000e+00
Sqrt of (4+8) = 3.464102e+00
Sqrt of (5+10) = 3.872983e+00
Sqrt of (6+12) = 4.242640e+00
Sqrt of (7+14) = 4.582576e+00
Sqrt of (8+16) = 4.898980e+00
Sqrt of (9+18) = 5.196152e+00
Sqrt of (10+20) = 5.477226e+00
Sqrt of (11+22) = 5.744563e+00
Sqrt of (12+24) = 6.000000e+00
Sqrt of (13+26) = 6.244998e+00
Sqrt of (14+28) = 6.480741e+00
Sqrt of (15+30) = 6.708204e+00
Sqrt of (16+32) = 6.928203e+00
Sqrt of (17+34) = 7.141428e+00
Sqrt of (18+36) = 7.348469e+00
Sqrt of (19+38) = 7.549834e+00
Sqrt of (20+40) = 7.745967e+00
Sqrt of (21+42) = 7.937254e+00
Sqrt of (22+44) = 8.124039e+00
Sqrt of (23+46) = 8.306623e+00
Sqrt of (24+48) = 8.485281e+00
Sqrt of (25+50) = 8.660254e+00
Sqrt of (26+52) = 8.831760e+00
Sqrt of (27+54) = 9.000000e+00
Sqrt of (28+56) = 9.165152e+00
Sqrt of (29+58) = 9.327379e+00
==375108== 
==375108== HEAP SUMMARY:
==375108==     in use at exit: 0 bytes in 0 blocks
==375108==   total heap usage: 4 allocs, 4 frees, 1,384 bytes allocated
==375108== 
==375108== All heap blocks were freed -- no leaks are possible
==375108== 
==375108== For lists of detected and suppressed errors, rerun with: -s
==375108== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
```

<a id="arrays_allocation_in_function"></a>
## 5 - Creating arrays on the heap with external functions

The fifth code used as part of this course; it serves to give a demonstration on:
* Writing a C code which creates a few simple arrays using malloc, and
* Does this memory allocation externally in a function.

Before running the commands below, make sure you are in the correct directory. From the repository root directory, type:

```bash
cd 5_Arrays_Functions
```

To build:
```bash
make
```
To execute:
```bash
./main.exe
```
To check for memory leaks with Valgrind:
```bash
valgrind ./main.exe
```

### Expected output

If executed with **"valgrind ./main.exe"** the expected output is identical to that shown in [the previous section](#heap_arrays).


<a id="save_fo_file"></a>
## 6 - Saving Results to File; Viewing them with Matplotlib

The sixth code used as part of this course; it serves to give a demonstration on:
* Writing a C code which creates some data and saves it to a tab delimited file, and
* Loads the file with a python script and creates an image of the data saved.

To build and execute:
```bash
make && ./main.exe
```

To load the data and create a graph using Matplotlib (Python):
```bash
python plot_results.py
```
### Expected output

Running the main.exe will produce a file (results.csv). The resulting graph generated by the **plot_results.py** script should look like:

![results.jpg](./6_Writing_Files/results.jpg)
