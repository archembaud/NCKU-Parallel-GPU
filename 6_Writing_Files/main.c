#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "memory.h"


void Save_Results(float *a, float *b, float *c, int N) {
    FILE *fptr;
    int i;
    fptr = fopen("results.csv", "w");
    // Write data to a file using tab delimiting
    for (i = 0; i < N; i++) {
        fprintf(fptr, "%g\t%g\t%g\n", a[i], b[i], c[i]);
    }
    // Close the file
    fclose(fptr);
}


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
        c[i] = sqrtf(a[i] + b[i]);
    }
    Save_Results(a, b, c, N);
    Free_Memory(&a, &b, &c);
    return 0;
}