/*
gpu.h
Declarations of functions used by gpu.cu
*/

void Allocate_Memory(float **h_a, float **h_b, float **d_a, float **d_b, int N);
void Free_Memory(float **h_a, float **h_b, float **d_a, float **d_b);
void Send_To_Device(float **h_a, float **d_a, int N);
void Get_From_Device(float **d_a, float **h_b, int N);
void Device_To_Device(float **d_dog, float **d_cat, int N);
void Compute_New_Temperature(float *d_a, float *d_b, float C, int N);
