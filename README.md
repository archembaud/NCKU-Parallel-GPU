# Introduction to Multi-Core CPU and GPU Compute
Codes used as part of the National Cheng-Kung University course on parallel and GPU compute within the department of mechanical engineering.

These codes can be subdivided into 4 categories:

## Basic C introduction codes

These codes are designed to give a basic overview of the C programming language, how these are compiled and built, and finally their execution in a standard linux environment.

[1 - Hello World](#hello_world)  
[2 - Simple Functions in C](#simple_functions)  
[3 - Arrays in C using the stack](#stack_arrays)

## Parallel processing using OpenMP


## Parallel processing using GPU (CUDA)


## Mathematics and Engineering Applications


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