/*
gpu.h
Declarations of functions used by gpu.cu
*/

// Define some constants
#define NX 100
#define NY 100
#define NZ 100
#define N (NX*NY*NZ)

#define L 0.05
#define W 0.05
#define H 0.03

#define DX (L/NX)
#define DY (W/NY)
#define DZ (H/NZ)

#define FIN_W 0.0015

#define DT 0.005
#define NO_STEPS 100000
#define K 16.25
#define RHO 8050.0
#define CP 502
#define ALPHA_X (DT*K/(RHO*CP*DX*DX))
#define ALPHA_Y (DT*K/(RHO*CP*DY*DY))
#define ALPHA_Z (DT*K/(RHO*CP*DZ*DZ))  
#define GPR 0.4991702  
#define AIR_RHO 1.25
#define AIR_VIS 1.81e-5
#define AIR_K 0.025
#define G 9.81

void Allocate_Memory(float **h_T, float **h_Body, float **d_T, float **d_Tnew, float **d_Body);
void Free_Memory(float **h_T, float **h_Body, float **d_T, float **d_Tnew, float **d_Body);
void Send_To_Device(float **h_a, float **d_a);
void Get_From_Device(float **d_a, float **h_b);
void Compute_T_new(float *d_T, float *d_Tnew, float *d_Body);
void Copy_on_Device(float **d_source, float **d_destination, int step);