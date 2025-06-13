/*
gpu.h
Declarations of functions used by gpu.cu
*/

void Allocate_Memory(float **h_temp, float **d_temp, float **d_temp_new, int N);
void Free_Memory(float **h_temp, float **d_temp, float **d_temp_new);
void Send_To_Device(float **h_a, float **d_a, const char *name, int N);
void Get_From_Device(float **d_a, float **h_b, const char *name, int N);
void Compute_Tnew(float *d_temp, float *d_temp_new, int N);
void Update_T(float **d_temp, float **d_temp_new, int N);
void Collect_Boundaries(float **d_temp, float **swap, int N, int tid);
void Distribute_Boundaries(float **d_temp, float **swap, int N, int tid);
