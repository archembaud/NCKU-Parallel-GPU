#include <stdio.h>
#include "gpu.h"

int main(int argc, char *argv[]) {

    float *h_a;
    float *d_a;
    int N = 100;

    // Allocate memory on both device and host
    Allocate_Memory(h_a, d_a, N);

    // Free memory
    Free_Memory(&h_a, &d_a);

    return 0;
}