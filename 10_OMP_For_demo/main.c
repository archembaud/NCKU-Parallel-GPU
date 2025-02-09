#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    // Create an array of 12 elements and record the thread assigned
    const int N = 12;
    int static_thread[N];
    int dynamic_thread[N];
    const int no_threads = 4;
    const int chunk = 2;
    int tid;
    // Set the number of OpenMP threads
    omp_set_num_threads(no_threads);

    // Create threads
    #pragma omp parallel private(tid) shared(static_thread, dynamic_thread)
    {

        // Use OMP for to iterate over the loops
        // Static scheduling forces the work across no_threads as
        // evenly as it can, regardless of whether or not this is faster.
        #pragma omp for schedule(static,chunk)
        for (int i = 0; i < N; i++) {
            tid = omp_get_thread_num();
            static_thread[i] = tid;
        }

        // Dynamic scheduling will allow whatever resource is free
        // and ready to pick the work up.
        #pragma omp for schedule(dynamic,chunk)
        for (int i = 0; i < N; i++) {
            tid = omp_get_thread_num();
            dynamic_thread[i] = tid;
        }

    } // Destroy threads

    // Show the resulting thread distribution
    for (int i = 0; i < N; i++) {
        printf("Static element thread[%d] = %d, Dynamic element thread[%d] = %d\n",
           i, static_thread[i], i, dynamic_thread[i]);
    }
    return 0;
}