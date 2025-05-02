/*
gpu.h
Declarations of functions used by gpu.cu
*/

void Allocate_Memory(float **h_T, float **h_Tnew, int **h_Body, float **d_T, float **d_Tnew, int **d_Body, int N);


void Free_Memory(float **h_T, float **h_Tnew, int **h_Body, float **d_T, float **d_Tnew, int **d_Body);


void Send_To_Device(float **h_T, float **d_T, int **h_Body, int **d_Body, int N);

void Get_From_Device(float **d_T, float **h_T, int N);