# HIP and ROCm

In recent times, computation on AMD GPU's is possible using an approach very similar to CUDA - through the use of ROCm.

## ROCm

ROCm is an Advanced Micro Devices (AMD) software stack for graphics processing unit (GPU) programming. ROCm spans several domains, including general-purpose computing on graphics processing units (GPGPU), high performance computing (HPC), and heterogeneous computing. It offers several programming models: HIP (GPU-kernel-based programming), OpenMP (directive-based programming), and OpenCL. The focus of the application of ROCm in this course is HIP.

## HIP

HIP -  Heterogeneous Interface for Portability - is a C++ Runtime API and Kernel Language that allows developers to create portable applications for AMD and NVIDIA GPUs from single source code. The compiler we used to generate executables for HIP is hipcc. The C++ code we write - and compile with hipcc - is very similar to CUDA:

* We allocate memory on the host and device in the same way,
* Memory is deliberately (by choice) copied between the host and device prior to, and following, computation,
* Kernel functions are used to define parallel work on the GPU; these operate using thread ID's and block ID's in the same way CUDA does.

Since the model is so similar, it is quite trivial to write codes which run on both Nvidia and AMD GPU devices.

## Examples and Demonstrations

We'll cover several computations using HIP / ROCm in this course:

* AMD_1_Hello_World: A simple Hello World code, which launches a very simple Kernel function.

* AMD_2_Vector_add: A more complete computation demonstration - a element-wise vector addition, written in the same style as the CUDA code previously shown.