#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    // Simple demonstration of single and critical
    int a, b;
    int tid;
    const int no_threads = 12;
    // Set the number of OpenMP threads
    omp_set_num_threads(no_threads);

    // Create threads
    #pragma omp parallel private(tid) shared(a, b)
    {
        tid = omp_get_thread_num();

        #pragma omp single
        {
            // Reset values for a and b using the first available worker
            printf("Thread %d resetting counts a and b\n", tid);            
            a = 0; b = 0;
        }
        
        #pragma omp critical
        {
            // Have each thread increment a one at a time (i.e. sequentially)
            printf("Thread %d incrementing a count\n", tid);
            a++;
        }

        #pragma omp master
        {
            // Have only the master thread (thread 0) increment b by 1
            printf("Thread %d incrementing b count\n", tid);
            b++;
        }
    } // Destroy threads

    printf("Values of a, b = %d, %d\n", a, b);
    return 0;
}