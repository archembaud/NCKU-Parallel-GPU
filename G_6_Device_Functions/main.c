#include <stdio.h>
#include "gpu.h"

int main(int argc, char *argv[]) {

    float *h_mass_flow_rate, *h_area, *h_Re;
    float *d_mass_flow_rate, *d_area, *d_Re;
    int N = 100;
    int i;

    // Allocate memory on both device and host
    Allocate_Memory(&h_mass_flow_rate, &h_area, &h_Re, &d_mass_flow_rate, &d_area, &d_Re, N);

    // Initialise mass flow rates and areas
    for (i = 0; i < N; i++) {
        h_mass_flow_rate[i] = 0.001*i + 0.1;  // kg/s
        h_area[i] = 0.0001*i + 0.01;          // m2
    }

    // Send both mass flow rate and area to the GPU
    Send_To_Device(&h_mass_flow_rate, &d_mass_flow_rate, "mass flow rates", N);
    Send_To_Device(&h_area, &d_area, "areas", N);

    // Compute the Reynolds number for each element on the GPU
    Compute_Re(d_mass_flow_rate, d_area, d_Re, N);

    // Copy the Reynolds numbers from the GPU to the host (CPU)
    Get_From_Device(&d_Re, &h_Re, "reynolds numbers", N);

    // Check the values of Reynolds numbers
    for (i = 0; i < N; i++) {
        printf("Value of h_Re[%d] = %g\n", i, h_Re[i]);
    }    

    // Free memory
    Free_Memory(&h_mass_flow_rate, &h_area, &h_Re, &d_mass_flow_rate, &d_area, &d_Re);

    return 0;
}