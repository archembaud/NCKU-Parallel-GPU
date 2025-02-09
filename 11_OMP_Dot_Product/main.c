#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    // Create two arrays (a,b) containing 64 elements
    // and then find the dot product of them.
    const int N = 64;
    float a[N]; float b[N];
    int no_threads = 4;
    float sum;

    // Initialise the value of a based on index
    // While we are here, let's compute the dot product
    sum = 0.0;
    for (int i = 0; i < N; i++) {
        a[i] = i; b[i] = 2.0*i;
        sum += a[i]*b[i];
    }
    printf("Dot product of a,b = %g\n", sum);

    // Set the number of OpenMP threads
    omp_set_num_threads(no_threads);
    sum = 0.0; // Reset the sum

    // Create threads and do the computation in parallel
    #pragma omp parallel shared(a, b, sum)
    {
        // Do the dot product first; we'll store the result in a[i]
        #pragma omp for
        for (int i = 0; i < N; i++) {
            a[i] = a[i]*b[i];
        }

        // Now compute the sum using an omp for reduction
        #pragma omp for reduction(+:sum)
        for (int i = 0; i < N; i++) {
            sum += a[i];
        }

    } // Destroy threads

    // Show the result
    printf("Dot product of a,b using openmp = %g\n", sum);
    return 0;
}