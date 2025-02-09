COMPILER := gcc
OPT_FLAGS := -O3 -lm
all: compile
	${COMPILER} main.o memory.o -o main.exe ${OPT_FLAGS}
compile:
	${COMPILER} memory.c -c ${OPT_FLAGS}
	${COMPILER} main.c -c ${OPT_FLAGS}
clean:
	rm main.exe main.o memory.o