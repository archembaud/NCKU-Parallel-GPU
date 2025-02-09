# NCKU-Parallel-GPU
Codes used as part of the National Cheng-Kung University course on parallel and GPU compute within the department of mechanical engineering.

These codes can be subdivided into 4 categories:

* Basic C introduction codes,
* Parallel processing using OpenMP,
* Parallel processing using GPU (CUDA), and
* Some mathematics and engineering application codes.


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