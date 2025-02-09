#include <stdio.h>
#include <math.h>

/*
Note - this example neglects the arguments provided with main;
Please see the example in 1_Hello_World for their application.
*/
int main(int argc, char *argv[]) {

    // Create an array on the stack
    const int N = 30;
    float a[N], b[N], c[N];
    int i;
    // Set the value of a and b for each element based on position
    for (i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = (float)2*i;
    }
    for (i = 0; i < N; i++) {
        c[i] = sqrtf(a[i] + b[i]);
        printf("Sqrt of (%g+%g) = %e\n", a[i], b[i], c[i]);
    }
    return 0;
}