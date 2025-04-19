COMPILER := gcc
OPT_FLAGS := -O3
all: compile
	${COMPILER} main.o -o main.exe ${OPT_FLAGS}
compile:
	${COMPILER} main.c -c ${OPT_FLAGS}
clean:
	rm main.exe main.o