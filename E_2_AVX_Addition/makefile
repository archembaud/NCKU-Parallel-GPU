all: AVX_MAIN
	gcc main.o memory.o -mavx -O3 -o main.exe
AVX_MAIN:
	gcc memory.c -O3 -c -mavx
	gcc main.c -O3 -c -mavx
clean:
	rm *.o *.exe