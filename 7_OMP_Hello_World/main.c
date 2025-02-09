#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[]) {
// Create threads
    #pragma omp parallel
    {
        printf("Hello!\n");
    } // Destroy threads
    return 0;
}