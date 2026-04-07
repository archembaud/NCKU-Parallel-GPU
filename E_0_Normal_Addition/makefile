all: SSE_MAIN
	gcc main.o memory.o -msse -fopenmp -ftree-vectorize -fopt-info-vec -O3 -o main.exe
SSE_MAIN:
	gcc memory.c -O3 -c -msse -fopenmp -ftree-vectorize -fopt-info-vec
	gcc main.c -O3 -c -msse -fopenmp -ftree-vectorize -fopt-info-vec
clean:
	rm *.o *.exe

