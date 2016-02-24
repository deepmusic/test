all: cpu_conv gpu_conv gpu_deconv gpu_proposal gpu_roipool gpu_pool
cpu_conv:
	gcc -std=c99 -o cpu_conv.o conv.c -lcblas
gpu_conv:
	nvcc -o gpu_conv.o conv.cu -lcublas -lcuda
gpu_deconv:
	nvcc -o gpu_deconv.o deconv.cu -lcublas -lcuda
gpu_proposal:
	nvcc -o gpu_proposal.o proposal.cu -lcuda
gpu_roipool:
	nvcc -o gpu_roipool.o roipool.cu -lcuda
gpu_pool:
	nvcc -o gpu_pool.o pool.cu -lcuda
gpu_relu:
	nvcc -o gpu_relu.o relu.cu -lcuda
