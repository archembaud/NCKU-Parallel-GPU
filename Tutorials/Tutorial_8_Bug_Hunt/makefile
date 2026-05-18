all: GPU CPU
	nvcc main.o gpu.o -code=sm_80 -arch=compute_80 -o main.exe
GPU:
	nvcc gpu.cu -c -code=sm_80 -arch=compute_80
CPU:
	g++ main.cpp -c
clean:
	rm *.o main.exe
