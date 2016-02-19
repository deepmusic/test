all: cpu_conv gpu_conv gpu_proposal
cpu_conv:
	gcc -std=c99 -o cpu_conv.o conv.c -lcblas
gpu_conv:
	nvcc -o gpu_conv.o conv.cu -lcublas -lcuda
gpu_proposal:
	nvcc -o gpu_proposal.o proposal.cu -lcublas -lcuda
