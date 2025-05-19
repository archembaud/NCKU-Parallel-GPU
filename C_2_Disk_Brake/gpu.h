/*
gpu.h
Declarations of functions used by gpu.cu
*/

// Define some constants
#define NX 1070
#define NY 1070
#define N (NX*NY)

#define L 0.18
#define W 0.18

#define DX (L/NX)
#define DY (W/NY)
#define DZ 0.002

#define DT 0.0001
#define NO_STEPS 12000
#define K 16.25
#define RHO 8050.0
#define CP 502
#define ALPHA_X (DT*K/(RHO*CP*DX*DX))
#define ALPHA_Y (DT*K/(RHO*CP*DY*DY))
  
#define AIR_RHO 1.25
#define AIR_VIS 1.81e-5
#define AIR_K 0.026


void Allocate_Memory(float **h_T, float **h_Body, float **d_T, float **d_Tnew, float **d_Body);
void Free_Memory(float **h_T, float **h_Body, float **d_T, float **d_Tnew, float **d_Body);
void Send_To_Device(float **h_a, float **d_a);
void Get_From_Device(float **d_a, float **h_b);
void Compute_T_new(float *d_T, float *d_Tnew, float *d_Body, int time_step);
void Copy_on_Device(float **d_source, float **d_destination, int step);