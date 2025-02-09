#include <stdio.h>
#include "sum_integers.h"

/*
Note - this example neglects the arguments provided with main;
Please see the example in 1_Hello_World for their application.
*/
int main(int argc, char *argv[]) {
    const int a = 1;
    const int b = -2;
    int c;
    c = sum_two_integers(a,b);
    printf("The value of %d + %d is %d\n", a, b, c);
    return 0;
}
