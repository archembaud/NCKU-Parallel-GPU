/*
gpu.h
Declarations of functions used by gpu.cu
*/

void Allocate_Memory(float **h_a, float **d_a, int N);
void Free_Memory(float **h_a, float **d_a);