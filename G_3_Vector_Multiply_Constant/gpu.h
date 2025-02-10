/*
gpu.h
Declarations of functions used by gpu.cu
*/

void Allocate_Memory(float **h_a, float **h_b, float **d_a, int N);
void Free_Memory(float **h_a, float **h_b, float **d_a);
void Send_To_Device(float **h_a, float **d_a, int N);
void Get_From_Device(float **d_a, float **h_b, int N);
void Vector_Times_Constant(float **d_a, float C, int N);