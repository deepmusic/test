CUDA_LIB_DIR := /usr/local/cuda/lib64
OPENCV_LIB_DIR := /usr/lib/x86_64-linux-gnu

INCLUDE_DIR := include
SRC_ROOT := src
BUILD_ROOT := build
MODULES := core layers nets outer util

SRC_DIR := $(addprefix $(SRC_ROOT)/, $(MODULES))
BUILD_DIR := $(addprefix $(BUILD_ROOT)/, $(MODULES))
SRC_FILES := $(foreach sdir, $(SRC_DIR), $(wildcard $(sdir)/*.cu))
CPU_OBJ_FILES := $(patsubst $(SRC_ROOT)/%.cu, $(BUILD_ROOT)/%.cpu.o, $(SRC_FILES))
GPU_OBJ_FILES := $(patsubst $(SRC_ROOT)/%.cu, $(BUILD_ROOT)/%.gpu.o, $(SRC_FILES))
INCLUDE_FLAG := $(addprefix -I, $(INCLUDE_DIR))

CC := g++
CFLAGS := -c -O3 -Wall -fPIC $(INCLUDE_FLAG)
LDFLAGS := -L$(OPENCV_LIB_DIR) -lcblas -lm -lopencv_core -lopencv_highgui
NVCC := /usr/local/cuda/bin/nvcc
NVCCFLAGS := -c -O3 -use_fast_math -Xcompiler '-fPIC -Wall -DGPU' $(INCLUDE_FLAG)
NVLDFLAGS := -L$(CUDA_LIB_DIR) -L$(OPENCV_LIB_DIR) -lcudart -lcublas -lopencv_core -lopencv_highgui

all: demo_cpu.bin demo_gpu.bin demo_cpu_compress.bin demo_gpu_compress.bin demo_cpu_light.bin demo_gpu_light.bin demo_cpu_light_compress.bin demo_gpu_light_compress.bin

# Runtime binary
demo_cpu.bin: $(BUILD_ROOT)/demo.cpu.test.o libdlcpu.so
	$(CC) -o $@ $^ $(LDFLAGS)
demo_gpu.bin: $(BUILD_ROOT)/demo.gpu.test.o libdlgpu.so
	$(NVCC) -o $@ $^ $(NVLDFLAGS)
demo_cpu_compress.bin: $(BUILD_ROOT)/demo.cpu.compress.test.o libdlcpu.so
	$(CC) -o $@ $^ $(LDFLAGS)
demo_gpu_compress.bin: $(BUILD_ROOT)/demo.gpu.compress.test.o libdlgpu.so
	$(NVCC) -o $@ $^ $(NVLDFLAGS)
demo_cpu_light.bin: $(BUILD_ROOT)/demo.cpu.light.test.o libdlcpu.so
	$(CC) -o $@ $^ $(LDFLAGS)
demo_gpu_light.bin: $(BUILD_ROOT)/demo.gpu.light.test.o libdlgpu.so
	$(NVCC) -o $@ $^ $(NVLDFLAGS)
demo_cpu_light_compress.bin: $(BUILD_ROOT)/demo.cpu.light.compress.test.o libdlcpu.so
	$(CC) -o $@ $^ $(LDFLAGS)
demo_gpu_light_compress.bin: $(BUILD_ROOT)/demo.gpu.light.compress.test.o libdlgpu.so
	$(NVCC) -o $@ $^ $(NVLDFLAGS)

# Shared library
libdlgpu.so: $(BUILD_DIR) $(GPU_OBJ_FILES) $(BUILD_ROOT)/link.gpu.o
	$(CC) -shared -o $@ $(GPU_OBJ_FILES) $(BUILD_ROOT)/link.gpu.o $(NVLDFLAGS)
libdlcpu.so: $(BUILD_DIR) $(CPU_OBJ_FILES)
	$(CC) -shared -o $@ $(CPU_OBJ_FILES) $(LDFLAGS)
$(BUILD_ROOT)/link.gpu.o: $(BUILD_DIR) $(GPU_OBJ_FILES)
	$(NVCC) -Xcompiler '-fPIC' -dlink $(GPU_OBJ_FILES) -o $@

# Static library
libdlcpu.a: $(BUILD_DIR) $(CPU_OBJ_FILES)
	ar rcs $@ $(CPU_OBJ_FILES)
libdlgpu.a: $(BUILD_DIR) $(GPU_OBJ_FILES)
	ar rcs $@ $(GPU_OBJ_FILES)

# Clean
clean:
	@rm -rf $(BUILD_DIR) $(BUILD_ROOT)
	@rm -rf *.so *.bin *.a *.pyc

vpath %.cu $(SRC_DIR) + $(SRC_ROOT)

$(BUILD_DIR):
	@mkdir -p $@

$(BUILD_ROOT)/%.cpu.test.o: %.cu
	@cp $< $<.cpp
	$(CC) $(CFLAGS) $<.cpp -o $@
	@rm -f $<.cpp
$(BUILD_ROOT)/%.cpu.compress.test.o: %.cu
	@cp $< $<.c.cpp
	$(CC) $(CFLAGS) -DCOMPRESS $<.c.cpp -o $@
	@rm -f $<.c.cpp
$(BUILD_ROOT)/%.cpu.light.test.o: %.cu
	@cp $< $<.l.cpp
	$(CC) $(CFLAGS) -DLIGHT_MODEL $<.l.cpp -o $@
	@rm -f $<.l.cpp
$(BUILD_ROOT)/%.cpu.light.compress.test.o: %.cu
	@cp $< $<.lc.cpp
	$(CC) $(CFLAGS) -DLIGHT_MODEL -DCOMPRESS $<.lc.cpp -o $@
	@rm -f $<.lc.cpp

$(BUILD_ROOT)/%.gpu.test.o: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@
$(BUILD_ROOT)/%.gpu.compress.test.o: %.cu
	$(NVCC) $(NVCCFLAGS) -DCOMPRESS $< -o $@
$(BUILD_ROOT)/%.gpu.light.test.o: %.cu
	$(NVCC) $(NVCCFLAGS) -DLIGHT_MODEL $< -o $@
$(BUILD_ROOT)/%.gpu.light.compress.test.o: %.cu
	$(NVCC) $(NVCCFLAGS) -DLIGHT_MODEL -DCOMPRESS $< -o $@

define make-goal
$1/%.cpu.o: %.cu
	@cp $$< $$<.c
	$(CC) $(CFLAGS) $$<.c -o $$@
	@rm -f $$<.c
$1/%.gpu.o: %.cu
	$(NVCC) $(NVCCFLAGS) $$< -o $$@
endef

$(foreach bdir, $(BUILD_DIR), $(eval $(call make-goal, $(bdir))))
