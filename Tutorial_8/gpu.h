/*
gpu.h
Declarations of functions used by gpu.cu
*/

#define a0 18.8e-6
#define a1 165e-6

void Set_GPU_Device(int device); 


void Allocate_Memory(float **h_T, float **h_Tnew, int **h_Body, float **d_T, float **d_Tnew, int **d_Body, int N);

void Compute_GPU_Crap(float *d_T, float *d_Tnew, int *d_Body, int NX, int NY, float DX, float DY, float DT);


void Compute_GPU_Good(float *d_T, float *d_Tnew, int *d_Body, int NX, int NY, float DX, float DY, float DT);


void Free_Memory(float **h_T, float **h_Tnew, int **h_Body, float **d_T, float **d_Tnew, int **d_Body);


void Send_To_Device(float **h_T, float **d_T, int **h_Body, int **d_Body, int N);

void Get_From_Device(float **d_T, float **h_T, int N);

void Update_Temperature_GPU(float **d_T, float **d_Tnew, int N);