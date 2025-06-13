all: GPU CPU
	g++ -fopenmp -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 main.o gpu.o -lcuda -lcudart -o main.exe
GPU:
	nvcc -code=sm_80 -arch=compute_80 -lcuda -lcudart  -L/usr/local/cuda/lib64 gpu.cu -c
CPU:
	g++ main.c -fopenmp -c -lcudart
clean:
	rm *.o main.exe
