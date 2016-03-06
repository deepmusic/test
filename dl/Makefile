CC := gcc
CFLAGS := -O3 -Wall -std=c99
LDFLAGS := -lcblas -lm
NVCC := nvcc
NVCCFLAGS := -O3 -use_fast_math -Xcompiler -Wall -DGPU
NVLDFLAGS := -lcuda -lcublas
BUILD_DIR := build

test_cpu: conv.cpu.test.bin deconv.cpu.test.bin dropout.cpu.test.bin fc.cpu.test.bin pool.cpu.test.bin proposal.nms.cpu.test.bin relu.cpu.test.bin roipool.cpu.test.bin
test_gpu: conv.gpu.test.bin deconv.gpu.test.bin dropout.gpu.test.bin fc.gpu.test.bin pool.gpu.test.bin proposal.nms.gpu.test.bin relu.gpu.test.bin roipool.gpu.test.bin

lib_cpu: tensor.cpu.o nms.cpu.o conv.cpu.o deconv.cpu.o fc.cpu.o dropout.cpu.o relu.cpu.o roipool.cpu.o pool.cpu.o proposal.cpu.o
	ar rcs $(BUILD_DIR)/libdlcpu.a $^
	rm $^
lib_gpu: tensor.gpu.o nms.gpu.o conv.gpu.o deconv.gpu.o fc.gpu.o dropout.gpu.o relu.gpu.o roipool.gpu.o pool.gpu.o proposal.gpu.o
	ar rcs $(BUILD_DIR)/libdlgpu.a $^
	rm $^

clean:
	rm -rf $(BUILD_DIR)

%.cpu.o: %.cu layer.h
	cp $< $*.c
	$(CC) $(CFLAGS) -c $*.c -o $@
	rm $*.c
%.cpu.test.o: %.cu layer.h
	cp $< $*.c
	$(CC) $(CFLAGS) -DTEST -c $*.c -o $@
	rm $*.c

%.gpu.o: %.cu layer.h cuda_settings.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
%.gpu.test.o: %.cu layer.h cuda_settings.h
	$(NVCC) $(NVCCFLAGS) -DTEST -c $< -o $@

%.cpu.test.bin: %.cpu.test.o tensor.cpu.o
	mkdir -p $(BUILD_DIR)
	$(CC) -o $(BUILD_DIR)/$@ $^ $(LDFLAGS)
%.nms.cpu.test.bin: %.cpu.test.o tensor.cpu.o nms.cpu.o
	mkdir -p $(BUILD_DIR)
	$(CC) -o $(BUILD_DIR)/$@ $^ $(LDFLAGS)

%.gpu.test.bin: %.gpu.test.o tensor.gpu.o
	mkdir -p $(BUILD_DIR)
	$(NVCC) -o $(BUILD_DIR)/$@ $^ $(NVLDFLAGS)
%.nms.gpu.test.bin: %.gpu.test.o tensor.gpu.o nms.gpu.o
	mkdir -p $(BUILD_DIR)
	$(NVCC) -o $(BUILD_DIR)/$@ $^ $(NVLDFLAGS)
