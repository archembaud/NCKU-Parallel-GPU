#include <stdio.h>
#include <stdlib.h>


#define a0 18.8e-6
#define a1 165e-6

void Set_GPU_Device(int device) {
    cudaSetDevice(device);
}

__global__  void Compute_New_Temperature_GPU_Good(float *d_T, float *d_Tnew, int *d_Body, int NX, int NY, float DX, float DY, float DT) {

    // Goal is to calculate h_Tnew
    int N = NX*NY;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        // We can use the index to compute the x, y cell.
        int xcell = (int)i/NY;
        int ycell = i - xcell*NY;

        float TC = d_T[i];
        float aC = (d_Body[i] == 0)*a0 + (d_Body[i] == 1)*a1;

        float TL, TR, TD, TU;
        float aL, aR, aD, aU;

        // Check right
        if (xcell == (NX-1)) {
            // We are on the right edge, its damn hot here.
            TR = 1000.0;
            aR = (d_Body[i] == 0)*a0 + (d_Body[i] == 1)*a1;
        } else {
            // We can check the cell to the right
            TR = d_T[i+NY];
            aR = (d_Body[i+NY] == 0)*a0 + (d_Body[i+NY] == 1)*a1;
        }
        // Check left
        if (xcell == 0) {
            // We are on the left edge, its cool here.
            TL = 300.0;
            aL = (d_Body[i] == 0)*a0 + (d_Body[i] == 1)*a1;
        } else {
            // We can check the cell to the right
            TL = d_T[i-NY];
            aL = (d_Body[i-NY] == 0)*a0 + (d_Body[i-NY] == 1)*a1;
        }
        // Vertical direction now
        // Check up (U)
        if (ycell == (NY-1)) {
            // We are on the top edge, no heat flow.
            TU = d_T[i];
            aU = (d_Body[i] == 0)*a0 + (d_Body[i] == 1)*a1;
        } else {
            // We can check the cell above
            TU = d_T[i+1];
            aU = (d_Body[i+1] == 0)*a0 + (d_Body[i+1] == 1)*a1;
        }
        // Check down (D)
        if (ycell == 0) {
            // We are on the bottom edge, its also insulated
            TD = d_T[i];
            aD = (d_Body[i] == 0)*a0 + (d_Body[i] == 1)*a1;
        } else {
            // We can check the cell to bottom
            TD = d_T[i-1];
            aD = (d_Body[i-1] == 0)*a0 + (d_Body[i-1] == 1)*a1;
        }

        // We are now ready to compute T_new
        d_Tnew[i] = d_T[i] + (2.0/( (1.0/aC) + (1.0/aR) ))*(DT/(DX*DX))*(TR - TC);
        d_Tnew[i] = d_Tnew[i] - (2.0/( (1.0/aC) + (1.0/aL) ))*(DT/(DX*DX))*(TC - TL);
        d_Tnew[i] = d_Tnew[i] + (2.0/( (1.0/aC) + (1.0/aU) ))*(DT/(DY*DY))*(TU - TC);
        d_Tnew[i] = d_Tnew[i] - (2.0/( (1.0/aC) + (1.0/aD) ))*(DT/(DY*DY))*(TC - TD);
    }

}


__global__  void Compute_New_Temperature_GPU_Crap(float *d_T, float *d_Tnew, int *d_Body, int NX, int NY, float DX, float DY, float DT) {

    // Goal is to calculate h_Tnew
    int N = NX*NY;

    for (int i = 0; i < N; i++) {
        // We can use the index to compute the x, y cell.
        int xcell = (int)i/NY;
        int ycell = i - xcell*NY;

        float TC = d_T[i];
        float aC = (d_Body[i] == 0)*a0 + (d_Body[i] == 1)*a1;

        float TL, TR, TD, TU;
        float aL, aR, aD, aU;

        // Check right
        if (xcell == (NX-1)) {
            // We are on the right edge, its damn hot here.
            TR = 1000.0;
            aR = (d_Body[i] == 0)*a0 + (d_Body[i] == 1)*a1;
        } else {
            // We can check the cell to the right
            TR = d_T[i+NY];
            aR = (d_Body[i+NY] == 0)*a0 + (d_Body[i+NY] == 1)*a1;
        }
        // Check left
        if (xcell == 0) {
            // We are on the left edge, its cool here.
            TL = 300.0;
            aL = (d_Body[i] == 0)*a0 + (d_Body[i] == 1)*a1;
        } else {
            // We can check the cell to the right
            TL = d_T[i-NY];
            aL = (d_Body[i-NY] == 0)*a0 + (d_Body[i-NY] == 1)*a1;
        }
        // Vertical direction now
        // Check up (U)
        if (ycell == (NY-1)) {
            // We are on the top edge, no heat flow.
            TU = d_T[i];
            aU = (d_Body[i] == 0)*a0 + (d_Body[i] == 1)*a1;
        } else {
            // We can check the cell above
            TU = d_T[i+1];
            aU = (d_Body[i+1] == 0)*a0 + (d_Body[i+1] == 1)*a1;
        }
        // Check down (D)
        if (ycell == 0) {
            // We are on the bottom edge, its also insulated
            TD = d_T[i];
            aD = (d_Body[i] == 0)*a0 + (d_Body[i] == 1)*a1;
        } else {
            // We can check the cell to bottom
            TD = d_T[i-1];
            aD = (d_Body[i-1] == 0)*a0 + (d_Body[i-1] == 1)*a1;
        }

        // We are now ready to compute T_new
        d_Tnew[i] = d_T[i] + (2.0/( (1.0/aC) + (1.0/aR) ))*(DT/(DX*DX))*(TR - TC);
        d_Tnew[i] = d_Tnew[i] - (2.0/( (1.0/aC) + (1.0/aL) ))*(DT/(DX*DX))*(TC - TL);
        d_Tnew[i] = d_Tnew[i] + (2.0/( (1.0/aC) + (1.0/aU) ))*(DT/(DY*DY))*(TU - TC);
        d_Tnew[i] = d_Tnew[i] - (2.0/( (1.0/aC) + (1.0/aD) ))*(DT/(DY*DY))*(TC - TD);
    }

}


// Write a simple wrapping function to call the crap GPU code
void Compute_GPU_Crap(float *d_T, float *d_Tnew, int *d_Body, int NX, int NY, float DX, float DY, float DT) {

    int threadsperblock = 1;
    int blockspergrid = 1;
    Compute_New_Temperature_GPU_Crap<<<blockspergrid,threadsperblock>>>(d_T, d_Tnew, d_Body, NX, NY, DX, DY, DT);

}

// Write a simple wrapping function to call the crap GPU code
void Compute_GPU_Good(float *d_T, float *d_Tnew, int *d_Body, int NX, int NY, float DX, float DY, float DT) {
    int N = NX*NY;
    int threadsperblock = 256;
    int blockspergrid = (N + threadsperblock - 1) / threadsperblock;
    Compute_New_Temperature_GPU_Good<<<blockspergrid,threadsperblock>>>(d_T, d_Tnew, d_Body, NX, NY, DX, DY, DT);

}

void Allocate_Memory(float **h_T, float **h_Tnew, int **h_Body, 
                     float **d_T, float **d_Tnew, int **d_Body, int N) {
    cudaError_t Error;
    // Host memory
    *h_T = (float*)malloc(N*sizeof(float));
    *h_Tnew = (float*)malloc(N*sizeof(float)); 
    *h_Body = (int*)malloc(N*sizeof(int)); 
    // Device memory
    Error = cudaMalloc((void**)d_T, N*sizeof(float));
    printf("CUDA error (malloc d_T) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_Tnew, N*sizeof(float));
    printf("CUDA error (malloc d_Tnew) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_Body, N*sizeof(int));
    printf("CUDA error (malloc d_Body) = %s\n", cudaGetErrorString(Error));
}

void Free_Memory(float **h_T, float **h_Tnew, int **h_Body, 
                 float **d_T, float **d_Tnew, int **d_Body) {
    if (*h_T) free(*h_T);
    if (*h_Tnew) free(*h_Tnew);
    if (*h_Body) free(*h_Body);
    if (*d_T) cudaFree(*d_T);
    if (*d_Tnew) cudaFree(*d_Tnew);
    if (*d_Body) cudaFree(*d_Body);
}

void Send_To_Device(float **h_T, float **d_T, int **h_Body, int **d_Body, int N) {
    // Grab an error type
    cudaError_t Error;
    // Send T to the GPU
    Error = cudaMemcpy(*d_T, *h_T, N*sizeof(float), cudaMemcpyHostToDevice); 
    printf("CUDA error (memcpy h_T -> d_T) = %s\n", cudaGetErrorString(Error));
    // Send Body to the GPU
    Error = cudaMemcpy(*d_Body, *h_Body, N*sizeof(float), cudaMemcpyHostToDevice); 
    printf("CUDA error (memcpy h_Body -> d_Body) = %s\n", cudaGetErrorString(Error));
}

void Get_From_Device(float **d_T, float **h_T, int N) {
    // Grab a error type
    cudaError_t Error;
    // Send d_a to the host variable h_b
    Error = cudaMemcpy(*h_T, *d_T, N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("CUDA error (memcpy d_T -> h_T) = %s\n", cudaGetErrorString(Error));
}

void Update_Temperature_GPU(float **d_T, float **d_Tnew, int N) {
    cudaError_t Error;
    // Send d_Tnew into d_T
    Error = cudaMemcpy(*d_T, *d_Tnew, N*sizeof(float), cudaMemcpyDeviceToDevice);
    //printf("CUDA error (memcpy d_Tnew -> d_T) = %s\n", cudaGetErrorString(Error));
}