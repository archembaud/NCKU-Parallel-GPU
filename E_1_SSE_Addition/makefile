all: SSE_MAIN
	gcc main.o memory.o -msse -O3 -o main.exe
SSE_MAIN:
	gcc memory.c -O3 -c -msse
	gcc main.c -O3 -c -msse
clean:
	rm *.o *.exe