/*
gpu.h
Declarations of functions used by gpu.cu
*/

void Set_GPU_Device(int device); 

void Allocate_Memory(float **h_T, float **h_Tnew, float **d_T, float **d_Tnew, int N);

void Compute_GPU(float *d_T, float *d_Tnew, int NX, int NY, float DX, float DY, float DT);

void Free_Memory(float **h_T, float **h_Tnew, float **d_T, float **d_Tnew);

void Send_To_Device(float **h_T, float **d_T, int N);

void Get_From_Device(float **d_T, float **h_T, int N);

void Update_Temperature_GPU(float **d_T, float **d_Tnew, int N);