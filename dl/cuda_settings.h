#ifndef PVA_DL_CUDA_SETTINGS_H
#define PVA_DL_CUDA_SETTINGS_H

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) printf("%s\n", cudaGetErrorString(error)); \
  } while (0)

#endif // PVA_DL_CUDA_SETTINGS_H
