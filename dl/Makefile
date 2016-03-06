all: cpu_conv gpu_conv cpu_deconv gpu_deconv cpu_fc gpu_fc cpu_pool gpu_pool cpu_roipool gpu_roipool cpu_relu gpu_relu cpu_proposal gpu_proposal cpu_dropout gpu_dropout
cpu_conv:
	cp conv.cu conv.c
	gcc -O3 -Wall -std=c99 -DTEST -o cpu_conv.o conv.c -lcblas
	rm conv.c
cpu_deconv:
	cp deconv.cu deconv.c
	gcc -O3 -Wall -std=c99 -DTEST -o cpu_deconv.o deconv.c -lcblas
	rm deconv.c
cpu_fc:
	cp fc.cu fc.c
	gcc -O3 -Wall -std=c99 -DTEST -o cpu_fc.o fc.c -lcblas
	rm fc.c
cpu_pool:
	cp pool.cu pool.c
	gcc -O3 -Wall -std=c99 -DTEST -o cpu_pool.o pool.c
	rm pool.c
cpu_roipool:
	cp roipool.cu roipool.c
	gcc -O3 -Wall -std=c99 -DTEST -o cpu_roipool.o roipool.c
	rm roipool.c
cpu_relu:
	cp relu.cu relu.c
	gcc -O3 -Wall -std=c99 -DTEST -o cpu_relu.o relu.c
	rm relu.c
cpu_proposal:
	cp proposal.cu proposal.c
	gcc -O3 -Wall -std=c99 -DTEST -o cpu_proposal.o proposal.c -lm
	rm proposal.c
cpu_dropout:
	cp dropout.cu dropout.c
	gcc -O3 -Wall -std=c99 -DTEST -o cpu_dropout.o dropout.c
	rm dropout.c
gpu_conv:
	nvcc -O3 -use_fast_math -Xcompiler -Wall -DGPU -DTEST -o gpu_conv.o conv.cu -lcublas -lcuda
gpu_deconv:
	nvcc -O3 -use_fast_math -Xcompiler -Wall -DGPU -DTEST -o gpu_deconv.o deconv.cu -lcublas -lcuda
gpu_fc:
	nvcc -O3 -use_fast_math -Xcompiler -Wall -DGPU -DTEST -o gpu_fc.o fc.cu -lcublas -lcuda
gpu_pool:
	nvcc -O3 -use_fast_math -Xcompiler -Wall -DGPU -DTEST -o gpu_pool.o pool.cu -lcuda
gpu_roipool:
	nvcc -O3 -use_fast_math -Xcompiler -Wall -DGPU -DTEST -o gpu_roipool.o roipool.cu -lcuda
gpu_relu:
	nvcc -O3 -use_fast_math -Xcompiler -Wall -DGPU -DTEST -o gpu_relu.o relu.cu -lcuda
gpu_proposal:
	nvcc -O3 -use_fast_math -Xcompiler -Wall -DGPU -DTEST -o gpu_proposal.o proposal.cu -lcuda
gpu_dropout:
	nvcc -O3 -use_fast_math -Xcompiler -Wall -DGPU -DTEST -o gpu_dropout.o dropout.cu -lcuda
