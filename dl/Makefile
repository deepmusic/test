all: cpu_conv gpu_conv cpu_deconv gpu_deconv cpu_roipool gpu_roipool cpu_relu gpu_relu gpu_proposal gpu_pool
cpu_conv:
	cp conv.cu conv.c
	gcc -O3 -Wall -std=c99 -o cpu_conv.o conv.c -lcblas
	rm conv.c
cpu_deconv:
	cp deconv.cu deconv.c
	gcc -O3 -Wall -std=c99 -o cpu_deconv.o deconv.c -lcblas
	rm deconv.c
cpu_roipool:
	cp roipool.cu roipool.c
	gcc -O3 -Wall -std=c99 -o cpu_roipool.o roipool.c
	rm roipool.c
cpu_relu:
	cp relu.cu relu.c
	gcc -O3 -Wall -std=c99 -o cpu_relu.o relu.c
	rm relu.c
gpu_conv:
	nvcc -O3 -use_fast_math -Xcompiler -Wall -DGPU -o gpu_conv.o conv.cu -lcublas -lcuda
gpu_deconv:
	nvcc -O3 -use_fast_math -Xcompiler -Wall -DGPU -o gpu_deconv.o deconv.cu -lcublas -lcuda
gpu_roipool:
	nvcc -O3 -use_fast_math -Xcompiler -Wall -DGPU -o gpu_roipool.o roipool.cu -lcuda
gpu_proposal:
	nvcc -O3 -use_fast_math -Xcompiler -Wall -DGPU -o gpu_proposal.o proposal.cu -lcuda
gpu_pool:
	nvcc -O3 -Xcompiler -Wall -o gpu_pool.o pool.cu -lcuda
gpu_relu:
	nvcc -O3 -use_fast_math -Xcompiler -Wall -DGPU -o gpu_relu.o relu.cu -lcuda
