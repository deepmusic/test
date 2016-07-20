CUDA_LIB_DIR := /usr/local/cuda/lib64
OPENCV_LIB_DIR := /usr/lib/x86_64-linux-gnu

INCLUDE_DIR := include
SRC_ROOT := src
BUILD_ROOT := build
MODULES := core layers nets outer

SRC_DIR := $(addprefix $(SRC_ROOT)/, $(MODULES))
BUILD_DIR := $(addprefix $(BUILD_ROOT)/, $(MODULES))
SRC_FILES := $(foreach sdir, $(SRC_DIR), $(wildcard $(sdir)/*.cu))
CPU_OBJ_FILES := $(patsubst $(SRC_ROOT)/%.cu, $(BUILD_ROOT)/%.cpu.o, $(SRC_FILES))
GPU_OBJ_FILES := $(patsubst $(SRC_ROOT)/%.cu, $(BUILD_ROOT)/%.gpu.o, $(SRC_FILES))
INCLUDE_FLAG := $(addprefix -I, $(INCLUDE_DIR))

CC := g++
CFLAGS := -c -O3 -Wall -fPIC $(INCLUDE_FLAG)
LDFLAGS := -L$(OPENCV_LIB_DIR) -lcblas -lm -lopencv_core -lopencv_highgui -lgomp
NVCC := /usr/local/cuda/bin/nvcc
NVCCFLAGS := -c -O3 -use_fast_math -Xcompiler '-fPIC -Wall -DGPU' $(INCLUDE_FLAG)
NVLDFLAGS := -L$(CUDA_LIB_DIR) -L$(OPENCV_LIB_DIR) -lcudart -lcublas -lopencv_core -lopencv_highgui

all: listdirs demo_cpu.bin demo_gpu.bin demo_cpu_compress.bin demo_gpu_compress.bin demo_cpu_light.bin demo_gpu_light.bin demo_cpu_light_compress.bin demo_gpu_light_compress.bin

# Runtime binary
demo_cpu.bin: $(BUILD_ROOT)/outer/demo.cpp.cpu.test.o libdlcpu.so
	$(CC) -o $@ $^ $(LDFLAGS)
demo_gpu.bin: $(BUILD_ROOT)/outer/demo.gpu.test.o libdlgpu.so
	$(NVCC) -o $@ $^ $(NVLDFLAGS)
demo_cpu_compress.bin: $(BUILD_ROOT)/outer/demo.cpp.cpu.compress.test.o libdlcpu.so
	$(CC) -o $@ $^ $(LDFLAGS)
demo_gpu_compress.bin: $(BUILD_ROOT)/outer/demo.gpu.compress.test.o libdlgpu.so
	$(NVCC) -o $@ $^ $(NVLDFLAGS)
demo_cpu_light.bin: $(BUILD_ROOT)/outer/demo.cpp.cpu.light.test.o libdlcpu.so
	$(CC) -o $@ $^ $(LDFLAGS)
demo_gpu_light.bin: $(BUILD_ROOT)/outer/demo.gpu.light.test.o libdlgpu.so
	$(NVCC) -o $@ $^ $(NVLDFLAGS)
demo_cpu_light_compress.bin: $(BUILD_ROOT)/outer/demo.cpp.cpu.light.compress.test.o libdlcpu.so
	$(CC) -o $@ $^ $(LDFLAGS)
demo_gpu_light_compress.bin: $(BUILD_ROOT)/outer/demo.gpu.light.compress.test.o libdlgpu.so
	$(NVCC) -o $@ $^ $(NVLDFLAGS)

# Shared library
libdlgpu.so: $(GPU_OBJ_FILES) $(BUILD_ROOT)/link.gpu.o
	$(CC) -shared -o $@ $^ $(NVLDFLAGS)
libdlcpu.so: $(CPU_OBJ_FILES)
	$(CC) -shared -o $@ $^ $(LDFLAGS)
$(BUILD_ROOT)/link.gpu.o: $(GPU_OBJ_FILES)
	$(NVCC) -Xcompiler '-fPIC' -dlink $^ -o $@

# Static library
libdlcpu.a: $(CPU_OBJ_FILES)
	ar rcs $@ $^
libdlgpu.a: $(GPU_OBJ_FILES)
	ar rcs $@ $^

# Clean
clean:
	@rm -rf $(BUILD_DIR) $(BUILD_ROOT)
	@rm -rf *.so *.bin *.a *.pyc

vpath %.cu $(SRC_DIR)

listdirs: $(BUILD_DIR)

$(BUILD_DIR):
	@mkdir -p $@

define make-goal
$1/%.cpu.o: %.cu
	@cp $$< $$*.c
	$(CC) $(CFLAGS) $$*.c -o $$@
	@rm -f $$*.c
$1/%.cpp.cpu.o: %.cu
	@cp $$< $$*.cpp
	$(CC) $(CFLAGS) $$*.cpp -o $$@
	@rm -f $$*.cpp
$1/%.cpu.test.o: %.cu
	@cp $$< $$*.c
	$(CC) $(CFLAGS) -DTEST $$*.c -o $$@
	@rm -f $$*.c
$1/%.cpp.cpu.test.o: %.cu
	@cp $$< $$*.cpp
	$(CC) $(CFLAGS) -DTEST $$*.cpp -o $$@
	@rm -f $$*.cpp
$1/%.cpp.cpu.compress.test.o: %.cu
	@cp $$< $$*.cpp
	$(CC) $(CFLAGS) -DTEST -DCOMPRESS $$*.cpp -o $$@
	@rm -f $$*.cpp
$1/%.cpp.cpu.light.test.o: %.cu
	@cp $$< $$*.cpp
	$(CC) $(CFLAGS) -DTEST -DLIGHT_MODEL $$*.cpp -o $$@
	@rm -f $$*.cpp
$1/%.cpp.cpu.light.compress.test.o: %.cu
	@cp $$< $$*.cpp
	$(CC) $(CFLAGS) -DTEST -DLIGHT_MODEL -DCOMPRESS $$*.cpp -o $$@
	@rm -f $$*.cpp

$1/%.gpu.o: %.cu
	$(NVCC) $(NVCCFLAGS) $$< -o $$@
$1/%.gpu.test.o: %.cu
	$(NVCC) $(NVCCFLAGS) -DTEST $$< -o $$@
$1/%.gpu.compress.test.o: %.cu
	$(NVCC) $(NVCCFLAGS) -DTEST -DCOMPRESS $$< -o $$@
$1/%.gpu.light.test.o: %.cu
	$(NVCC) $(NVCCFLAGS) -DTEST -DLIGHT_MODEL $$< -o $$@
$1/%.gpu.light.compress.test.o: %.cu
	$(NVCC) $(NVCCFLAGS) -DTEST -DLIGHT_MODEL -DCOMPRESS $$< -o $$@
endef

$(foreach bdir, $(BUILD_DIR), $(eval $(call make-goal, $(bdir))))
