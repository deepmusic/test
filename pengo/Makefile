CC := g++
CFLAGS := -c -O3 -Wall
NVCC := /usr/local/cuda/bin/nvcc
NVCCFLAGS := -O3 -use_fast_math -Xcompiler '-fPIC -Wall -DGPU'
CUDA_LIB_DIR := /usr/local/cuda/lib64
OPENCV_LIB_DIR := /usr/lib/x86_64-linux-gnu
LDFLAGS := -L$(OPENCV_LIB_DIR) -lcblas -lm -lopencv_core -lopencv_highgui
NVLDFLAGS := -L$(CUDA_LIB_DIR) -L$(OPENCV_LIB_DIR) -lcudart -lcublas -lopencv_core -lopencv_highgui

all: test_gpu test_cpu

test_gpu: test_pvanet.main.o libdlgpu.so
	$(CC) -o $@ $^ $(NVLDFLAGS)
	rm $<

test_cpu: test_pvanet.main.o libdlcpu.so
	$(CC) -o $@ $^ $(LDFLAGS)
	rm $<

libdlgpu.so: link.o tensor.gpu.o net.gpu.o nms.gpu.o softmax.gpu.o conv.gpu.o deconv.gpu.o fc.gpu.o dropout.gpu.o relu.gpu.o roipool.gpu.o pool.gpu.o proposal.gpu.o concat.gpu.o odout.gpu.o image.gpu.o frcnn.gpu.o pvanet.o
	$(CC) -shared -o $@ $^ $(NVLDFLAGS)
	rm $^

libdlcpu.so: tensor.cpu.o net.cpu.o nms.cpu.o softmax.cpu.o conv.cpu.o deconv.cpu.o fc.cpu.o dropout.cpu.o relu.cpu.o roipool.cpu.o pool.cpu.o proposal.cpu.o concat.cpu.o odout.cpu.o image.cpu.o frcnn.cpu.o pvanet.o
	$(CC) -shared -o $@ $^ $(LDFLAGS)
	rm $^

clean:
	rm -f *.o test_gpu test_cpu libdlgpu.so libdlcpu.so

link.o: tensor.gpu.o net.gpu.o nms.gpu.o softmax.gpu.o conv.gpu.o deconv.gpu.o fc.gpu.o dropout.gpu.o relu.gpu.o roipool.gpu.o pool.gpu.o proposal.gpu.o concat.gpu.o odout.gpu.o image.gpu.o frcnn.gpu.o pvanet.o
	$(NVCC) -Xcompiler '-fPIC' -dlink $^ -o $@

%.main.o: %.cpp pvanet.hpp net.h
	$(CC) $(CFLAGS) $< -o $@

pvanet.o: pvanet.cpp pvanet.hpp net.h
	$(CC) $(CFLAGS) -fPIC $< -o $@

%.gpu.o: %.cu layer.h cuda_settings.h net.h
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

%.cpu.o: %.cu layer.h
	cp $< $*.c
	$(CC) $(CFLAGS) $*.c -fPIC -o $@
	rm $*.c
