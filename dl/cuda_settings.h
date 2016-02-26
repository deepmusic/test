#ifndef PVA_DL_CUDA_SETTINGS_H
#define PVA_DL_CUDA_SETTINGS_H

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>

#include <stdio.h>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      printf("[CUDA ERROR] %s\n", cudaGetErrorString(error)); \
    } \
  } while (0)

#endif // endifndef PVA_DL_CUDA_SETTINGS_H
