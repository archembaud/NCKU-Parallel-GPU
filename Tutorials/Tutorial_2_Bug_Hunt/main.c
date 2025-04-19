#include <stdio.h>

int Add_Two_Variables(int *A, int *B, int *C) {
    for (int i = 0; i <= 10; i++) {
        C[i] = A[i] + B[i];
    }
    return 0;
}


int main() {
    int a[10];
    int b[10];
    int c[10];
    int result;
    // Set b and c
    for (int i = 0; i <= 10; i++) {
        b[i] = i;
        c[i] = 2*i;
    }
    // Compute a = b + c
    result = Add_Two_Variables(a,b,c);
    // Print out a
    // Should be:
    // a[0] = 0
    // a[1] = 3
    // a[2] = 6
    // a[3] = 9
    // a[4] = 12
    // a[5] = 15
    // a[6] = 18
    // a[7] = 21
    // a[8] = 24
    // a[9] = 27
    for (int i = 0; i <= 10; i++) {
        printf("a[%d] = %d\n", i, a[i]);
    }

    return result;
}