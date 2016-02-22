all: cpu_conv gpu_conv gpu_proposal gpu_roipool
cpu_conv:
	gcc -std=c99 -o cpu_conv.o conv.c -lcblas
gpu_conv:
	nvcc -o gpu_conv.o conv.cu -lcublas -lcuda
gpu_proposal:
	nvcc -o gpu_proposal.o proposal.cu -lcuda
gpu_roipool:
	nvcc -o gpu_roipool.o roipool.cu -lcuda
