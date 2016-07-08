CUDA_LIB_DIR := /usr/local/cuda/lib64
OPENCV_LIB_DIR := /usr/lib/x86_64-linux-gnu

CC := g++
CFLAGS := -c -O3 -Wall -fPIC
LDFLAGS := -L$(OPENCV_LIB_DIR) -lcblas -lm -lopencv_core -lopencv_highgui -lgomp
NVCC := /usr/local/cuda/bin/nvcc
NVCCFLAGS := -c -O3 -use_fast_math -Xcompiler '-fPIC -Wall -DGPU'
NVLDFLAGS := -L$(CUDA_LIB_DIR) -L$(OPENCV_LIB_DIR) -lcudart -lcublas -lopencv_core -lopencv_highgui

all: test_gpu.bin test_cpu.bin

test_cpu.bin: test_pvanet.cpp.cpu.test.o pvanet.cpp.cpu.o libdlcpu.so
	$(CC) -o $@ $^ $(LDFLAGS)
	rm *.cpp.cpu.o
test_gpu.bin: test_pvanet.cpp.cpu.test.o pvanet.cpp.cpu.o libdlgpu.so
	$(CC) -o $@ $^ $(NVLDFLAGS)
	rm *.cpp.cpu.o

demo_cpu.bin: demo.cpp.cpu.test.o libdlcpu.so
	$(CC) -o $@ $^ $(LDFLAGS)
	rm $<
demo_gpu.bin: demo.gpu.test.o libdlgpu.so
	$(NVCC) -o $@ $^ $(NVLDFLAGS)
	rm $<

loader.bin: loader.cpu.o tensor.cpu.o net.cpu.o
	$(CC) -o $@ $^ $(LDFLAGS)
	rm $<

libdlgpu.so: tensor.gpu.o net.gpu.o nms.gpu.o scale.gpu.o crelu.gpu.o eltwise.gpu.o softmax.gpu.o conv.gpu.o deconv.gpu.o fc.gpu.o dropout.gpu.o relu.gpu.o roipool.gpu.o pool.gpu.o proposal.gpu.o concat.gpu.o odout.gpu.o odtest.gpu.o image.gpu.o frcnn.gpu.o wrapper.gpu.o link.gpu.o
	$(CC) -shared -o $@ $^ $(NVLDFLAGS)
	rm $^
libdlcpu.so: tensor.cpu.o net.cpu.o nms.cpu.o scale.cpu.o crelu.cpu.o eltwise.cpu.o softmax.cpu.o conv.cpu.o deconv.cpu.o fc.cpu.o dropout.cpu.o relu.cpu.o roipool.cpu.o pool.cpu.o proposal.cpu.o concat.cpu.o odout.cpu.o odtest.cpu.o image.cpu.o frcnn.cpu.o wrapper.cpu.o
	$(CC) -shared -o $@ $^ $(LDFLAGS)
	rm $^
link.gpu.o: tensor.gpu.o net.gpu.o nms.gpu.o scale.gpu.o crelu.gpu.o eltwise.gpu.o softmax.gpu.o conv.gpu.o deconv.gpu.o fc.gpu.o dropout.gpu.o relu.gpu.o roipool.gpu.o pool.gpu.o proposal.gpu.o concat.gpu.o odout.gpu.o odtest.gpu.o image.gpu.o frcnn.gpu.o wrapper.gpu.o
	$(NVCC) -Xcompiler '-fPIC' -dlink $^ -o $@

libdlcpu.a: tensor.cpu.o net.cpu.o nms.cpu.o scale.cpu.o crelu.cpu.o eltwise.cpu.o softmax.cpu.o conv.cpu.o deconv.cpu.o fc.cpu.o dropout.cpu.o relu.cpu.o roipool.cpu.o pool.cpu.o proposal.cpu.o concat.cpu.o odout.cpu.o odtest.cpu.o image.cpu.o frcnn.cpu.o wrapper.cpu.o
	ar rcs $@ $^
	rm $^
libdlgpu.a: tensor.gpu.o net.gpu.o nms.gpu.o scale.gpu.o crelu.gpu.o eltwise.gpu.o softmax.gpu.o conv.gpu.o deconv.gpu.o fc.gpu.o dropout.gpu.o relu.gpu.o roipool.gpu.o pool.gpu.o proposal.gpu.o concat.gpu.o odout.gpu.o odtest.gpu.o image.gpu.o frcnn.gpu.o wrapper.gpu.o
	ar rcs $@ $^
	rm $^

clean:
	rm -f *.cpu.o *.gpu.o *.bin *.test.o *.so *.a *.pyc

conv.cpu.o: conv.cu layer.h
	cp $< $*.cpp
	$(CC) $(CFLAGS) -fopenmp $*.cpp -o $@
	rm $*.cpp

%.cpu.o: %.cu layer.h
	cp $< $*.c
	$(CC) $(CFLAGS) $*.c -o $@
	rm $*.c
%.cpp.cpu.o: %.cu layer.h
	cp $< $*.cpp
	$(CC) $(CFLAGS) $*.cpp -o $@
	rm $*.cpp
%.cpu.test.o: %.cu layer.h
	cp $< $*.c
	$(CC) $(CFLAGS) -DTEST $*.c -o $@
	rm $*.c
%.cpp.cpu.test.o: %.cu layer.h
	cp $< $*.cpp
	$(CC) $(CFLAGS) -DTEST $*.cpp -o $@
	rm $*.cpp

%.gpu.o: %.cu layer.h
	$(NVCC) $(NVCCFLAGS) $< -o $@
%.gpu.test.o: %.cu layer.h
	$(NVCC) $(NVCCFLAGS) -DTEST $< -o $@
