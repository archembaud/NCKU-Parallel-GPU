#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "memory.h"


int main(int argc, char *argv[]) {
    const int N = 30;
    float *a, *b, *c;
    int i;
    int error;
    error = Allocate_Memory(&a, &b, &c, N);    
    if (error == 1) {
        printf("Aborting main program.\n");
        return 1;
    }
    // Set the value of a and b for each element based on position
    for (i = 0; i < N; i++) {
        a[i] = (float)i; b[i] = (float)2*i;
    }
    for (i = 0; i < N; i++) {
        c[i] = sqrtf(a[i] + b[i]);
        printf("Sqrt of (%g+%g) = %e\n", a[i], b[i], c[i]);
    }
    Free_Memory(&a, &b, &c);
    return 0;
}