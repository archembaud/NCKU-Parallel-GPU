all: GPU CPU
	nvcc main.o gpu.o -arch=compute_80 -o main.exe
GPU:
	nvcc gpu.cu -c -arch=compute_80 
CPU:
	g++ main.c -c
clean:
	rm *.o main.exe