/*
gpu.h
Declarations of functions used by gpu.cu
*/

void Allocate_Memory(float **h_a, float **d_a, const char *variable, const int N);
void Free_Memory(float **h_a, float **d_a);
void Send_To_Device(float **h_a, float **d_a, const char *variable, const int N);
void Get_From_Device(float **d_a, float **h_b, const char *variable, const int N);
void Partial_Dot_Product(float **d_a, float **d_b, float **d_c, const int N);