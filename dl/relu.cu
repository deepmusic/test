#include "layer.h"

#ifdef GPU
#include "cuda_settings.h"
#endif

// relu transform bottom -> top
//   top[i] = 0 if bottom[i] <= 0
#ifdef GPU
__global__
void relu_gpu(const real* const bottom, real* const top,
              const int data_size)
{
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < data_size;
       index += blockDim.x) {
    if (bottom[index] > 0) {
      top[index] = bottom[index];
    }
    else {
      top[index] = 0;
    }
  }
}
#else
void relu_cpu(const real* const bottom, real* const top,
              const int data_size)
{
  for (int index = 0; index < data_size; ++index) {
    if (bottom[index] > 0) {
      top[index] = bottom[index];
    }
    else {
      top[index] = 0;
    }
  }
}
#endif

// soft relu transform bottom -> top
//   top[i] = slope * bottom[i] if bottom[i] <= 0
#ifdef GPU
__global__
void prelu_gpu(const real* const bottom, real* const top,
               const int data_size, const real negative_slope)
{
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < data_size;
       index += blockDim.x) {
    if (bottom[index] > 0) {
      top[index] = bottom[index];
    }
    else {
      top[index] = negative_slope * bottom[index];
    }
  }
}
#else
void prelu_cpu(const real* const bottom, real* const top,
               const int data_size, const real negative_slope)
{
  for (int index = 0; index < data_size; ++index) {
    if (bottom[index] > 0) {
      top[index] = bottom[index];
    }
    else {
      top[index] = negative_slope * bottom[index];
    }
  }
}
#endif

// in-place relu transform
#ifdef GPU
__global__
void relu_inplace_gpu(real* const bottom, const int data_size)
{
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < data_size;
       index += blockDim.x) {
    if (bottom[index] < 0) {
      bottom[index] = 0;
    }
  }
}
#else
void relu_inplace_cpu(real* const bottom, const int data_size)
{
  for (int index = 0; index < data_size; ++index) {
    if (bottom[index] < 0) {
      bottom[index] = 0;
    }
  }
}
#endif

// in-place soft relu transform
#ifdef GPU
__global__
void prelu_inplace_gpu(real* const bottom, const int data_size,
                       const real negative_slope)
{
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < data_size;
       index += blockDim.x) {
    if (bottom[index] < 0) {
      bottom[index] *= negative_slope;
    }
  }
}
#else
void prelu_inplace_cpu(real* const bottom, const int data_size,
                       const real negative_slope)
{
  for (int index = 0; index < data_size; ++index) {
    if (bottom[index] < 0) {
      bottom[index] *= negative_slope;
    }
  }
}
#endif

// relu transform: bottom -> top
//   data size: whole size (N * C * H * W or something)
void relu_forward(const Tensor* const bottom, Tensor* const top,
                  const ReluOption* const option)
{
  const int data_size = flatten_size(bottom);

#ifdef GPU
  const int threads_per_block = 512;
  const int num_blocks = DIV_THEN_CEIL(data_size, threads_per_block);
  if (option->negative_slope == 0) {
    relu_gpu<<<num_blocks, threads_per_block>>>(
        bottom->data, top->data, data_size);
  }
  else {
    prelu_gpu<<<num_blocks, threads_per_block>>>(
        bottom->data, top->data, data_size, option->negative_slope);
  }
#else
  if (option->negative_slope == 0) {
    relu_cpu(
        bottom->data, top->data, data_size);
  }
  else {
    prelu_cpu(
        bottom->data, top->data, data_size, option->negative_slope);
  }
#endif

  top->ndim = bottom->ndim;
  top->num_items = bottom->num_items;
  for (int n = 0; n < bottom->num_items; ++n) {
    for (int i = 0; i < bottom->ndim; ++i) {
      top->shape[n][i] = bottom->shape[n][i];
    }
  }
}

// in-place relu transform: bottom -> bottom
//   data size: whole size (N * C * H * W or something)
void relu_forward_inplace(Tensor* const bottom,
                          const ReluOption* const option)
{
  const int data_size = flatten_size(bottom);

#ifdef GPU
  const int threads_per_block = 512;
  const int num_blocks = DIV_THEN_CEIL(data_size, threads_per_block);
  if (option->negative_slope == 0) {
    relu_inplace_gpu<<<num_blocks, threads_per_block>>>(
        bottom->data, data_size);
  }
  else {
    prelu_inplace_gpu<<<num_blocks, threads_per_block>>>(
        bottom->data, data_size, option->negative_slope);
  }
#else
  if (option->negative_slope == 0) {
    relu_inplace_cpu(
        bottom->data, data_size);
  }
  else {
    prelu_inplace_cpu(
        bottom->data, data_size, option->negative_slope);
  }
#endif
}

#include <stdio.h>
#include <stdlib.h>

#define DATA_SIZE 384*18*23

int main(int argc, char **argv)
{
  Tensor X, Y;
  real* X_data = (real*)malloc(DATA_SIZE * sizeof(real));
  real* Y_data = (real*)malloc(DATA_SIZE * sizeof(real));
  ReluOption option;

  {
    option.negative_slope = 0;
  }

  {
    X.ndim = 3;
    X.num_items = 1;
    for (int i = 0; i < X.num_items; ++i) {
      X.shape[i][0] = 384;
      X.shape[i][1] = 18;
      X.shape[i][2] = 23;
    }
  }
 
  {
    FILE* fp;
    int X_size = flatten_size(&X);

    printf("data loading\n");

    fp = fopen("../data/temp/conv_top0.bin", "rb");
    if ((int)fread(X_data, sizeof(real), X_size, fp) != X_size) {
      printf("Error while reading conv_top0\n");
    }
    fclose(fp);
  }

#ifdef GPU
  {
    printf("set device\n");
    CUDA_CHECK(cudaSetDevice(0));
  }
#endif

#ifdef GPU
  {
    int X_size = flatten_size(&X);

    printf("cuda malloc\n");
    CUDA_CHECK(cudaMalloc(&X.data, X_size*sizeof(real)));
    CUDA_CHECK(cudaMalloc(&Y.data, X_size*sizeof(real)));

    printf("memcopy\n");
    CUDA_CHECK(cudaMemcpy(X.data, X_data, X_size*sizeof(real), cudaMemcpyHostToDevice));
  }
#else
  {
    X.data = X_data;
    Y.data = Y_data;
  }
#endif

  {
    printf("do forward\n");
    relu_forward(&X, &Y, &option);
  }

#ifdef GPU
  {
    int Y_size = flatten_size(&Y);
    printf("memcpy\n");
    CUDA_CHECK(cudaMemcpy(Y_data, Y.data, Y_size*sizeof(real), cudaMemcpyDeviceToHost));
  }
#endif

  {
    int Y_size = flatten_size(&Y);
    for (int i = 0; i < Y_size; ++i) {
      if (Y_data[i] != X_data[i] && (Y_data[i] != 0 || X_data[i] > 0)) {
        printf("top[%d] = %.6f, bottom[%d] = %.6f\n", i, Y_data[i], i, X_data[i]);
      }
    }
  }

  {
    printf("free\n");
    free(X_data);
    free(Y_data);
  }
#ifdef GPU
  {
    printf("cuda free\n");
    CUDA_CHECK(cudaFree(X.data));
    CUDA_CHECK(cudaFree(Y.data));
  }
#endif

  return 0;
}
