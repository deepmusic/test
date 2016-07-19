CUDA_LIB_DIR := /usr/local/cuda/lib64
OPENCV_LIB_DIR := /usr/lib/x86_64-linux-gnu

CC := g++
CFLAGS := -c -O3 -Wall -fPIC
LDFLAGS := -L$(OPENCV_LIB_DIR) -lcblas -lm -lopencv_core -lopencv_highgui -lgomp
NVCC := /usr/local/cuda/bin/nvcc
NVCCFLAGS := -c -O3 -use_fast_math -Xcompiler '-fPIC -Wall -DGPU'
NVLDFLAGS := -L$(CUDA_LIB_DIR) -L$(OPENCV_LIB_DIR) -lcudart -lcublas -lopencv_core -lopencv_highgui

all: demo_cpu.bin demo_gpu.bin demo_cpu_compress.bin demo_gpu_compress.bin demo_cpu_light.bin demo_gpu_light.bin demo_cpu_light_compress.bin demo_gpu_light_compress.bin

# Runtime binary
demo_cpu.bin: demo.cpp.cpu.test.o libdlcpu.so
	$(CC) -o $@ $^ $(LDFLAGS)
	rm $<
demo_gpu.bin: demo.gpu.test.o libdlgpu.so
	$(NVCC) -o $@ $^ $(NVLDFLAGS)
	rm $<
demo_cpu_compress.bin: demo.cpp.cpu.compress.test.o libdlcpu.so
	$(CC) -o $@ $^ $(LDFLAGS)
	rm $<
demo_gpu_compress.bin: demo.gpu.compress.test.o libdlgpu.so
	$(NVCC) -o $@ $^ $(NVLDFLAGS)
	rm $<
demo_cpu_light.bin: demo.cpp.cpu.light.test.o libdlcpu.so
	$(CC) -o $@ $^ $(LDFLAGS)
	rm $<
demo_gpu_light.bin: demo.gpu.light.test.o libdlgpu.so
	$(NVCC) -o $@ $^ $(NVLDFLAGS)
	rm $<
demo_cpu_light_compress.bin: demo.cpp.cpu.light.compress.test.o libdlcpu.so
	$(CC) -o $@ $^ $(LDFLAGS)
	rm $<
demo_gpu_light_compress.bin: demo.gpu.light.compress.test.o libdlgpu.so
	$(NVCC) -o $@ $^ $(NVLDFLAGS)
	rm $<

# Caffe model loader (WIP)
loader.bin: loader.cpu.o tensor.cpu.o net.cpu.o
	$(CC) -o $@ $^ $(LDFLAGS)
	rm $<

# Shared library
libdlgpu.so: tensor.gpu.o net.gpu.o nms.gpu.o scale.gpu.o crelu.gpu.o eltwise.gpu.o softmax.gpu.o reshape.gpu.o conv.gpu.o deconv.gpu.o fc.gpu.o dropout.gpu.o relu.gpu.o roipool.gpu.o pool.gpu.o proposal.gpu.o concat.gpu.o odout.gpu.o image.gpu.o frcnn.gpu.o cnn.gpu.o cnn_light.gpu.o wrapper.gpu.o link.gpu.o
	$(CC) -shared -o $@ $^ $(NVLDFLAGS)
	rm $^
libdlcpu.so: tensor.cpu.o net.cpu.o nms.cpu.o scale.cpu.o crelu.cpu.o eltwise.cpu.o softmax.cpu.o reshape.cpu.o conv.cpu.o deconv.cpu.o fc.cpu.o dropout.cpu.o relu.cpu.o roipool.cpu.o pool.cpu.o proposal.cpu.o concat.cpu.o odout.cpu.o image.cpu.o frcnn.cpu.o cnn.cpu.o cnn_light.cpu.o wrapper.cpu.o
	$(CC) -shared -o $@ $^ $(LDFLAGS)
	rm $^
link.gpu.o: tensor.gpu.o net.gpu.o nms.gpu.o scale.gpu.o crelu.gpu.o eltwise.gpu.o softmax.gpu.o reshape.gpu.o conv.gpu.o deconv.gpu.o fc.gpu.o dropout.gpu.o relu.gpu.o roipool.gpu.o pool.gpu.o proposal.gpu.o concat.gpu.o odout.gpu.o image.gpu.o frcnn.gpu.o cnn.gpu.o cnn_light.gpu.o wrapper.gpu.o
	$(NVCC) -Xcompiler '-fPIC' -dlink $^ -o $@

# Static library
libdlcpu.a: tensor.cpu.o net.cpu.o nms.cpu.o scale.cpu.o crelu.cpu.o eltwise.cpu.o softmax.cpu.o reshape.cpu.o conv.cpu.o deconv.cpu.o fc.cpu.o dropout.cpu.o relu.cpu.o roipool.cpu.o pool.cpu.o proposal.cpu.o concat.cpu.o odout.cpu.o image.cpu.o frcnn.cpu.o cnn.cpu.o cnn_light.cpu.o wrapper.cpu.o
	ar rcs $@ $^
	rm $^
libdlgpu.a: tensor.gpu.o net.gpu.o nms.gpu.o scale.gpu.o crelu.gpu.o eltwise.gpu.o softmax.gpu.o reshape.cpu.o conv.gpu.o deconv.gpu.o fc.gpu.o dropout.gpu.o relu.gpu.o roipool.gpu.o pool.gpu.o proposal.gpu.o concat.gpu.o odout.gpu.o image.gpu.o frcnn.gpu.o cnn.gpu.o cnn_light.gpu.o wrapper.gpu.o
	ar rcs $@ $^
	rm $^

clean:
	rm -f *.cpu.o *.gpu.o *.bin *.test.o *.so *.a *.pyc

conv.cpu.o: conv.cu layer.h
	cp $< $*.cpp
	$(CC) $(CFLAGS) -fopenmp $*.cpp -o $@
	rm -f $*.cpp

%.cpu.o: %.cu layer.h
	cp $< $*.c
	$(CC) $(CFLAGS) $*.c -o $@
	rm -f $*.c
%.cpp.cpu.o: %.cu layer.h
	cp $< $*.cpp
	$(CC) $(CFLAGS) $*.cpp -o $@
	rm -f $*.cpp

%.gpu.o: %.cu layer.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

%.cpu.test.o: %.cu layer.h
	cp $< $*.c
	$(CC) $(CFLAGS) -DTEST $*.c -o $@
	rm -f $*.c
%.cpp.cpu.test.o: %.cu layer.h
	cp $< $*.cpp
	$(CC) $(CFLAGS) -DTEST $*.cpp -o $@
	rm -f $*.cpp
%.cpp.cpu.compress.test.o: %.cu layer.h
	cp $< $*.cpp
	$(CC) $(CFLAGS) -DTEST -DCOMPRESS $*.cpp -o $@
	rm -f $*.cpp
%.cpp.cpu.light.test.o: %.cu layer.h
	cp $< $*.cpp
	$(CC) $(CFLAGS) -DTEST -DLIGHT_MODEL $*.cpp -o $@
	rm -f $*.cpp
%.cpp.cpu.light.compress.test.o: %.cu layer.h
	cp $< $*.cpp
	$(CC) $(CFLAGS) -DTEST -DLIGHT_MODEL -DCOMPRESS $*.cpp -o $@
	rm -f $*.cpp

%.gpu.test.o: %.cu layer.h
	$(NVCC) $(NVCCFLAGS) -DTEST $< -o $@
%.gpu.compress.test.o: %.cu layer.h
	$(NVCC) $(NVCCFLAGS) -DTEST -DCOMPRESS $< -o $@
%.gpu.light.test.o: %.cu layer.h
	$(NVCC) $(NVCCFLAGS) -DTEST -DLIGHT_MODEL $< -o $@
%.gpu.light.compress.test.o: %.cu layer.h
	$(NVCC) $(NVCCFLAGS) -DTEST -DLIGHT_MODEL -DCOMPRESS $< -o $@
