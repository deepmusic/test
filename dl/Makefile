CC := gcc
CFLAGS := -O3 -Wall -std=c99
NVCC := nvcc
NVCCFLAGS := -O3 -use_fast_math -Xcompiler -Wall -DGPU
BUILD_DIR := build

test_cpu: test_cpu_conv test_cpu_deconv test_cpu_fc test_cpu_dropout test_cpu_relu test_cpu_roipool test_cpu_pool test_cpu_proposal
test_gpu: test_gpu_conv test_gpu_deconv test_gpu_fc test_gpu_dropout test_gpu_relu test_gpu_roipool test_gpu_pool test_gpu_proposal

clean:
	rm -rf $(BUILD_DIR)

$(BUILD_DIR)/%.cpu.o: %.cu layer.h
	mkdir -p $(BUILD_DIR)
	cp $< $*.c
	$(CC) $(CFLAGS) -c $*.c
	mv $*.o $@
	rm $*.c
$(BUILD_DIR)/%.cpu.test.o: %.cu layer.h
	mkdir -p $(BUILD_DIR)
	cp $< $*.c
	$(CC) $(CFLAGS) -DTEST -c $*.c
	mv $*.o $@
	rm $*.c

$(BUILD_DIR)/%.gpu.o: %.cu layer.h
	$(NVCC) $(NVCCFLAGS) -c $<
	mv $*.o $@
$(BUILD_DIR)/%.gpu.test.o: %.cu layer.h
	$(NVCC) $(NVCCFLAGS) -DTEST -c $<
	mv $*.o $@

$(BUILD_DIR)/%.cpu.test: $(BUILD_DIR)/%.cpu.test.o $(BUILD_DIR)/tensor.cpu.o
	$(CC) -o $@ $< $(BUILD_DIR)/tensor.cpu.o
$(BUILD_DIR)/%.blas.cpu.test: $(BUILD_DIR)/%.cpu.test.o $(BUILD_DIR)/tensor.cpu.o
	$(CC) -o $@ $< $(BUILD_DIR)/tensor.cpu.o -lcblas
$(BUILD_DIR)/%.nms.cpu.test: $(BUILD_DIR)/%.cpu.test.o $(BUILD_DIR)/tensor.cpu.o $(BUILD_DIR)/nms.cpu.o
	$(CC) -o $@ $< $(BUILD_DIR)/tensor.cpu.o $(BUILD_DIR)/nms.cpu.o -lm

$(BUILD_DIR)/%.gpu.test: $(BUILD_DIR)/%.gpu.test.o $(BUILD_DIR)/tensor.gpu.o
	$(NVCC) -o $@ $< $(BUILD_DIR)/tensor.gpu.o -lcuda
$(BUILD_DIR)/%.blas.gpu.test: $(BUILD_DIR)/%.gpu.test.o $(BUILD_DIR)/tensor.gpu.o
	$(NVCC) -o $@ $< $(BUILD_DIR)/tensor.gpu.o -lcuda -lcublas
$(BUILD_DIR)/%.nms.gpu.test: $(BUILD_DIR)/%.gpu.test.o $(BUILD_DIR)/tensor.gpu.o $(BUILD_DIR)/nms.gpu.o
	$(NVCC) -o $@ $< $(BUILD_DIR)/tensor.gpu.o $(BUILD_DIR)/nms.gpu.o -lcuda

test_cpu_conv: $(BUILD_DIR)/conv.blas.cpu.test
	mv $< $(BUILD_DIR)/test_cpu_conv.o
test_cpu_deconv: $(BUILD_DIR)/deconv.blas.cpu.test
	mv $< $(BUILD_DIR)/test_cpu_deconv.o
test_cpu_fc: $(BUILD_DIR)/fc.blas.cpu.test
	mv $< $(BUILD_DIR)/test_cpu_fc.o
test_cpu_dropout: $(BUILD_DIR)/dropout.cpu.test
	mv $< $(BUILD_DIR)/test_cpu_dropout.o
test_cpu_relu: $(BUILD_DIR)/relu.cpu.test
	mv $< $(BUILD_DIR)/test_cpu_relu.o
test_cpu_roipool: $(BUILD_DIR)/roipool.cpu.test
	mv $< $(BUILD_DIR)/test_cpu_roipool.o
test_cpu_pool: $(BUILD_DIR)/pool.cpu.test
	mv $< $(BUILD_DIR)/test_cpu_pool.o
test_cpu_proposal: $(BUILD_DIR)/proposal.nms.cpu.test
	mv $< $(BUILD_DIR)/test_cpu_proposal.o

test_gpu_conv: $(BUILD_DIR)/conv.blas.gpu.test
	mv $< $(BUILD_DIR)/test_gpu_conv.o
test_gpu_deconv: $(BUILD_DIR)/deconv.blas.gpu.test
	mv $< $(BUILD_DIR)/test_gpu_deconv.o
test_gpu_fc: $(BUILD_DIR)/fc.blas.gpu.test
	mv $< $(BUILD_DIR)/test_gpu_fc.o
test_gpu_dropout: $(BUILD_DIR)/dropout.gpu.test
	mv $< $(BUILD_DIR)/test_gpu_dropout.o
test_gpu_relu: $(BUILD_DIR)/relu.gpu.test
	mv $< $(BUILD_DIR)/test_gpu_relu.o
test_gpu_roipool: $(BUILD_DIR)/roipool.gpu.test
	mv $< $(BUILD_DIR)/test_gpu_roipool.o
test_gpu_pool: $(BUILD_DIR)/pool.gpu.test
	mv $< $(BUILD_DIR)/test_gpu_pool.o
test_gpu_proposal: $(BUILD_DIR)/proposal.nms.gpu.test
	mv $< $(BUILD_DIR)/test_gpu_proposal.o
