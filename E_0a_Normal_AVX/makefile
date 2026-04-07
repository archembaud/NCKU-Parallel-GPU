all: AVX_MAIN
	gcc main.o memory.o -fopenmp -march=native -mavx -O3 -ftree-vectorize -fopt-info-vec -o main.exe
AVX_MAIN:
	gcc memory.c -O3 -fopenmp -march=native -ftree-vectorize -fopt-info-vec -c -mavx
	gcc main.c -O3 -fopenmp -march=native -ftree-vectorize -fopt-info-vec -c -mavx
clean:
	rm *.o *.exe
