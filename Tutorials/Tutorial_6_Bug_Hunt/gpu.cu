#include <stdio.h>
#include <stdlib.h>

void Allocate_Memory(float **h_a, float **d_a, int N) {
	size_t size = N*sizeof(float);
	cudaError_t Error;
	// Host memory
	*h_a = (float*)malloc(size); 
	// Device memory
	Error = cudaMalloc((void**)d_a, size); 
    printf("CUDA error (malloc d_a) = %d\n", cudaGetErrorString(Error));
}

void Free_Memory(float **h_a, float **d_a) {
	if (*h_a) free(*h_a);
	if (*d_a) free(*d_a);
}