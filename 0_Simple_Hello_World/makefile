COMPILER := gcc
OPT_FLAGS := -O3
all: preprocess compile
	${COMPILER} main.o -o main.exe ${OPT_FLAGS}
preprocess:
	${COMPILER} -E main.c -o main.i
compile:
	${COMPILER} -S main.i -o main.s
	${COMPILER} main.c -c ${OPT_FLAGS}
clean:
	rm main.exe main.o