COMPILER := gcc
OPT_FLAGS := -O3
all: compile
	${COMPILER} main.o sum_integers.o -o main.exe ${OPT_FLAGS}
compile:
	${COMPILER} sum_integers.c -c ${OPT_FLAGS}
	${COMPILER} main.c -c ${OPT_FLAGS}

clean:
	rm main.exe main.o