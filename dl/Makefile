CC := g++
CFLAGS := -O3 -Wall -std=c99
LDFLAGS := -lcblas -lm -lopencv_core -lopencv_highgui
NVCC := /usr/local/cuda/bin/nvcc
NVCCFLAGS := -O3 -use_fast_math -Xcompiler -Wall -DGPU
NVLDFLAGS := -lcuda -lcublas -lopencv_core -lopencv_highgui
BUILD_DIR := build

frcnn_gpu.bin: frcnn.gpu.o $(BUILD_DIR)/libdlgpu.a
	$(NVCC) -o $@ $^ $(NVLDFLAGS)
frcnn_cpu.bin: frcnn.cpp.cpu.o $(BUILD_DIR)/libdlcpu.a
	$(CC) -o $@ $^ $(LDFLAGS)

test_cpu: conv.cpu.test.bin deconv.cpu.test.bin dropout.cpu.test.bin fc.cpu.test.bin pool.cpu.test.bin proposal.nms.cpu.test.bin relu.cpu.test.bin roipool.cpu.test.bin
test_gpu: conv.gpu.test.bin deconv.gpu.test.bin dropout.gpu.test.bin fc.gpu.test.bin pool.gpu.test.bin proposal.nms.gpu.test.bin relu.gpu.test.bin roipool.gpu.test.bin

$(BUILD_DIR)/libdlcpu.a: tensor.cpu.o nms.cpu.o softmax.cpu.o conv.cpu.o deconv.cpu.o fc.cpu.o dropout.cpu.o relu.cpu.o roipool.cpu.o pool.cpu.o proposal.cpu.o concat.cpu.o odout.cpu.o image.cpp.cpu.o
	mkdir -p $(BUILD_DIR)
	ar rcs $@ $^
	#rm $^
$(BUILD_DIR)/libdlgpu.a: tensor.gpu.o nms.gpu.o softmax.gpu.o conv.gpu.o deconv.gpu.o fc.gpu.o dropout.gpu.o relu.gpu.o roipool.gpu.o pool.gpu.o proposal.gpu.o concat.gpu.o odout.gpu.o image.gpu.o
	mkdir -p $(BUILD_DIR)
	ar rcs $@ $^
	#rm $^

clean:
	rm -rf $(BUILD_DIR)
	rm *.cpu.o *.gpu.o *.bin

%.cpu.o: %.cu layer.h
	cp $< $*.c
	$(CC) $(CFLAGS) -c $*.c -o $@
	rm $*.c
%.cpp.cpu.o: %.cu layer.h
	cp $< $*.cpp
	$(CC) $(CFLAGS) -c $*.cpp -o $@
	rm $*.cpp
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
