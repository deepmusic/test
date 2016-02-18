all: cpu gpu
cpu:
	gcc -std=c99 -o cpu.o conv.c -lcblas
gpu:
	nvcc -o gpu.o conv.cu -lcublas -lcuda
