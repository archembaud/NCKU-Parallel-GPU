all: GPU CPU
	nvcc main.o gpu.o -architecture=compute_80 -o main.exe
GPU:
	nvcc gpu.cu -c -architecture=compute_80 
CPU:
	g++ main.c -o
clean:
	rm *.o main.exe