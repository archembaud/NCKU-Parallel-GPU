/*
gpu.h
Declarations of functions used by gpu.cu
*/

void Allocate_Memory(float **h_mass_flow_rate, float **h_area, float **h_Re, float **d_mass_flow_rate, float **d_area, float **d_Re, int N);
void Free_Memory(float **h_mass_flow_rate, float **h_area, float **h_Re, float **d_mass_flow_rate, float **d_area, float **d_Re);
void Send_To_Device(float **h_a, float **d_a, const char *name, int N);
void Get_From_Device(float **d_a, float **h_b, const char *name, int N);
void Compute_Re(float *d_mass_flow_rate, float *d_area, float *d_Re, int N);