CUDA_LIB_DIR := /usr/local/cuda/lib64
OPENCV_LIB_DIR := /usr/lib/x86_64-linux-gnu

CC := g++
CFLAGS := -c -O3 -Wall -fPIC
LDFLAGS := -L$(OPENCV_LIB_DIR) -lcblas -lm -lopencv_core -lopencv_highgui -lgomp
NVCC := /usr/local/cuda/bin/nvcc
NVCCFLAGS := -c -O3 -use_fast_math -Xcompiler '-fPIC -Wall -DGPU'
NVLDFLAGS := -L$(CUDA_LIB_DIR) -L$(OPENCV_LIB_DIR) -lcudart -lcublas -lopencv_core -lopencv_highgui

all: test_gpu.bin test_cpu.bin

test_cpu.bin: test_pvanet.cpp.cpu.o pvanet.cpp.cpu.o libdlcpu.so
	$(CC) -o $@ $^ $(LDFLAGS)
	rm $<
test_gpu.bin: test_pvanet.cpp.cpu.o pvanet.cpp.cpu.o libdlgpu.so
	$(CC) -o $@ $^ $(NVLDFLAGS)
	rm $<

demo_cpu.bin: demo.cpp.cpu.test.o libdlcpu.so
	$(CC) -o $@ $^ $(LDFLAGS)
demo_gpu.bin: demo.cpp.gpu.test.o libdlgpu.so
	$(NVCC) -o $@ $^ $(NVLDFLAGS)

frcnn_cpu.bin: frcnn.cpp.cpu.test.o libdlcpu.so
	$(CC) -o $@ $^ $(LDFLAGS)
frcnn_gpu.bin: frcnn.cpp.gpu.test.o libdlgpu.so
	$(NVCC) -o $@ $^ $(NVLDFLAGS)

libdlgpu.so: tensor.gpu.o net.cpp.gpu.o nms.gpu.o softmax.gpu.o conv.cpp.gpu.o deconv.cpp.gpu.o fc.cpp.gpu.o dropout.gpu.o relu.cpp.gpu.o roipool.cpp.gpu.o pool.cpp.gpu.o proposal.cpp.gpu.o concat.cpp.gpu.o odout.gpu.o odtest.gpu.o image.cpp.gpu.o frcnn.cpp.gpu.o logger.cpp.gpu.o link.gpu.o
	$(CC) -shared -o $@ $^ $(NVLDFLAGS)
	rm $^
libdlcpu.so: tensor.cpu.o net.cpp.cpu.o nms.cpu.o softmax.cpu.o conv.cpp.cpu.o deconv.cpp.cpu.o fc.cpp.cpu.o dropout.cpu.o relu.cpp.cpu.o roipool.cpp.cpu.o pool.cpp.cpu.o proposal.cpp.cpu.o concat.cpp.cpu.o odout.cpu.o odtest.cpu.o image.cpp.cpu.o frcnn.cpp.cpu.o logger.cpp.cpu.o
	$(CC) -shared -o $@ $^ $(LDFLAGS)
	rm $^

libdlcpu.a: tensor.cpu.o net.cpu.o nms.cpu.o softmax.cpu.o conv.cpu.o deconv.cpu.o fc.cpu.o dropout.cpu.o relu.cpu.o roipool.cpu.o pool.cpu.o proposal.cpu.o concat.cpu.o odout.cpu.o odtest.cpu.o image.cpp.cpu.o frcnn.cpu.o
	ar rcs $@ $^
	rm $^
libdlgpu.a: tensor.gpu.o net.gpu.o nms.gpu.o softmax.gpu.o conv.gpu.o deconv.gpu.o fc.gpu.o dropout.gpu.o relu.gpu.o roipool.gpu.o pool.gpu.o proposal.gpu.o concat.gpu.o odout.gpu.o odtest.gpu.o image.gpu.o frcnn.gpu.o
	ar rcs $@ $^
	rm $^

unit_test_cpu: conv.cpu.test.bin deconv.cpu.test.bin dropout.cpu.test.bin fc.cpu.test.bin pool.cpu.test.bin proposal.nms.cpu.test.bin relu.cpu.test.bin roipool.cpu.test.bin
unit_test_gpu: conv.gpu.test.bin deconv.gpu.test.bin dropout.gpu.test.bin fc.gpu.test.bin pool.gpu.test.bin proposal.nms.gpu.test.bin relu.gpu.test.bin roipool.gpu.test.bin

clean:
	rm -f *.cpu.o *.gpu.o *.bin *.test.o *.so *.a

conv.cpp.cpu.o: conv.cpp layer.h
	$(CC) $(CFLAGS) -fopenmp $< -o $@

%.cpu.o: %.c layer.h
	$(CC) $(CFLAGS) $< -o $@
%.cpp.cpu.o: %.cpp layer.h
	$(CC) $(CFLAGS) $< -o $@
%.cpu.test.o: %.cu layer.h
	cp $< $*.c
	$(CC) $(CFLAGS) -DTEST $*.c -o $@
	rm $*.c
%.cpp.cpu.test.o: %.cpp layer.h
	$(CC) $(CFLAGS) -DTEST $< -o $@

link.gpu.o: tensor.gpu.o net.cpp.gpu.o nms.gpu.o softmax.gpu.o conv.cpp.gpu.o deconv.cpp.gpu.o fc.cpp.gpu.o dropout.gpu.o relu.cpp.gpu.o roipool.cpp.gpu.o pool.cpp.gpu.o proposal.cpp.gpu.o concat.cpp.gpu.o odout.gpu.o odtest.gpu.o image.cpp.gpu.o frcnn.cpp.gpu.o logger.cpp.gpu.o
	$(NVCC) -Xcompiler '-fPIC' -dlink $^ -o $@

%.gpu.o: %.c layer.h
	cp $< $*.cu
	$(NVCC) $(NVCCFLAGS) $*.cu -o $@
	rm $*.cu
%.cpp.gpu.o: %.cpp layer.h
	cp $< $*.cu
	$(NVCC) $(NVCCFLAGS) $*.cu -o $@
	rm $*.cu
%.gpu.test.o: %.c layer.h
	cp $< $*.cu
	$(NVCC) $(NVCCFLAGS) -DTEST $*.cu -o $@
	cp $< $*.cu
%.cpp.gpu.test.o: %.cpp layer.h
	cp $< $*.cu
	$(NVCC) $(NVCCFLAGS) -DTEST $*.cu -o $@
	cp $< $*.cu

%.cpu.test.bin: %.cpu.test.o tensor.cpu.o
	$(CC) -o $@ $^ $(LDFLAGS)
%.nms.cpu.test.bin: %.cpu.test.o tensor.cpu.o nms.cpu.o
	$(CC) -o $@ $^ $(LDFLAGS)

%.gpu.test.bin: %.gpu.test.o tensor.gpu.o
	$(NVCC) -o $@ $^ $(NVLDFLAGS)
%.nms.gpu.test.bin: %.gpu.test.o tensor.gpu.o nms.gpu.o
	$(NVCC) -o $@ $^ $(NVLDFLAGS)
