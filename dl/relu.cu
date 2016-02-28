#include "layer.h"

// --------------------------------------------------------------------------
// kernel code
//   relu_{gpu, cpu}
//   prelu_{gpu, cpu}
//   relu_inplace_{gpu, cpu}
//   prelu_inplace_{gpu, cpu}
// --------------------------------------------------------------------------

// ReLU transform bottom -> top
//   top[i] = 0 if bottom[i] <= 0
#ifdef GPU
__global__
void relu_gpu(const real* const bottom, real* const top,
              const int data_size)
{
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < data_size;
       index += blockDim.x) {
    top[index] = (bottom[index] > 0) * bottom[index];
  }
}
#else
void relu_cpu(const real* const bottom, real* const top,
              const int data_size)
{
  for (int index = 0; index < data_size; ++index) {
    top[index] = (bottom[index] > 0) * bottom[index];
  }
}
#endif

// soft ReLU transform bottom -> top
//   top[i] = slope * bottom[i] if bottom[i] <= 0
#ifdef GPU
__global__
void prelu_gpu(const real* const bottom, real* const top,
               const int data_size, const real negative_slope)
{
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < data_size;
       index += blockDim.x) {
    top[index] = bottom[index] * ((bottom[index] > 0)
                                  + (bottom[index] <= 0) * negative_slope);
  }
}
#else
void prelu_cpu(const real* const bottom, real* const top,
               const int data_size, const real negative_slope)
{
  for (int index = 0; index < data_size; ++index) {
    top[index] = bottom[index] * ((bottom[index] > 0)
                                  + (bottom[index] <= 0) * negative_slope);
  }
}
#endif

// in-place ReLU transform
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

// in-place soft ReLU transform
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



// --------------------------------------------------------------------------
// layer operator code
//   relu_forward
//   relu_forward_inplace
// --------------------------------------------------------------------------

// (soft-)ReLU transform: bottom -> top
//   data size: total number of nodes (N * C * H * W or something)
//   if option->negative_slope = 0, perform ReLU
//                             > 0, perform soft ReLU
void relu_forward(const Tensor* const bottom,
                  Tensor* const top,
                  const ReluOption* const option)
{
  const int data_size = flatten_size(bottom);

  // perform (soft-)ReLU transform
  //   if option->negative_slope = 0, perform ReLU
  //                             > 0, perform soft ReLU
  #ifdef GPU
  {
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
  }
  #else
  {
    if (option->negative_slope == 0) {
      relu_cpu(
          bottom->data, top->data, data_size);
    }
    else {
      prelu_cpu(
          bottom->data, top->data, data_size, option->negative_slope);
    }
  }
  #endif

  // set top shape (= bottom shape)
  {
    top->ndim = bottom->ndim;
    top->num_items = bottom->num_items;
    for (int n = 0; n < bottom->num_items; ++n) {
      for (int i = 0; i < bottom->ndim; ++i) {
        top->shape[n][i] = bottom->shape[n][i];
      }
    }
  }
}

// in-place (soft-)ReLU transform: bottom -> bottom
//   data size: total number of nodes (N * C * H * W or something)
//   if option->negative_slope = 0, perform ReLU
//                             > 0, perform soft ReLU
void relu_forward_inplace(Tensor* const bottom,
                          const ReluOption* const option)
{
  const int data_size = flatten_size(bottom);

  // perform (soft-)ReLU transform
  //   if option->negative_slope = 0, perform ReLU
  //                             > 0, perform soft ReLU
  #ifdef GPU
  {
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
  }
  #else
  {
    if (option->negative_slope == 0) {
      relu_inplace_cpu(
          bottom->data, data_size);
    }
    else {
      prelu_inplace_cpu(
          bottom->data, data_size, option->negative_slope);
    }
  }
  #endif
}



// --------------------------------------------------------------------------
// test code
// --------------------------------------------------------------------------

#ifdef TEST
#include <stdio.h>
#include <stdlib.h>

#define DATA_SIZE 384*18*23

int main(int argc, char *argv[])
{
  // variable declaration & memory allocation
  Tensor X, Y_relu, Y_prelu;
  real* const X_data = (real*)malloc(DATA_SIZE * sizeof(real));
  real* const relu_data = (real*)malloc(DATA_SIZE * sizeof(real));
  real* const prelu_data = (real*)malloc(DATA_SIZE * sizeof(real));
  ReluOption option;

  // set option
  {
    option.negative_slope = 0;
  }

  // set data shapes
  {
    X.ndim = 3;
    X.num_items = 1;
    for (int i = 0; i < X.num_items; ++i) {
      X.shape[i][0] = 384;
      X.shape[i][1] = 18;
      X.shape[i][2] = 23;
    }
  }
 
  // load data
  {
    FILE* fp;
    const int X_size = flatten_size(&X);

    printf("data loading\n");

    fp = fopen("../data/temp/conv_top0.bin", "rb");
    if ((int)fread(X_data, sizeof(real), X_size, fp) != X_size) {
      printf("Error while reading conv_top0\n");
    }
    fclose(fp);
  }

  // CUDA initialization
  #ifdef GPU
  {
    printf("set device\n");
    CUDA_CHECK(cudaSetDevice(0));
  }
  #endif

  // bind loaded data to corresponding tensors
  #ifdef GPU
  {
    const int X_size = flatten_size(&X);

    printf("gpu malloc\n");
    CUDA_CHECK(cudaMalloc(&X.data, X_size*sizeof(real)));
    CUDA_CHECK(cudaMalloc(&Y_relu.data, X_size*sizeof(real)));
    CUDA_CHECK(cudaMalloc(&Y_prelu.data, X_size*sizeof(real)));

    printf("memcpy: cpu -> gpu\n");
    CUDA_CHECK(cudaMemcpy(X.data, X_data, X_size*sizeof(real),
                          cudaMemcpyHostToDevice));
  }
  #else
  {
    X.data = X_data;
    Y_relu.data = relu_data;
    Y_prelu.data = prelu_data;
  }
  #endif

  // do forward operation
  {
    printf("do forward (relu)\n");
    relu_forward(&X, &Y_relu, &option);

    printf("do forward (prelu)\n");
    option.negative_slope = 0.1f;
    relu_forward(&X, &Y_prelu, &option);
  }

  // copy GPU data to main memory
  #ifdef GPU
  {
    const int Y_size = flatten_size(&X);

    printf("memcpy: cpu <- gpu (relu)\n");
    CUDA_CHECK(cudaMemcpy(relu_data, Y_relu.data, Y_size*sizeof(real),
                          cudaMemcpyDeviceToHost));

    printf("memcpy: cpu <- gpu (prelu)\n");
    CUDA_CHECK(cudaMemcpy(prelu_data, Y_prelu.data, Y_size*sizeof(real),
                          cudaMemcpyDeviceToHost));
  }
  #endif

  // verify results
  {
    const int Y_size = flatten_size(&X);

    printf("verification (relu)\n");

    for (int i = 0; i < Y_size; ++i) {
      if (relu_data[i] != X_data[i]
          && (relu_data[i] != 0 || X_data[i] > 0)) {
        printf("top[%d] = %.6f, bottom[%d] = %.6f\n",
               i, relu_data[i], i, X_data[i]);
      }
    }

    printf("verification (prelu)\n");

    for (int i = 0; i < Y_size; ++i) {
      if (prelu_data[i] != X_data[i]
          && (prelu_data[i] != option.negative_slope * X_data[i]
              || X_data[i] > 0)) {
        printf("top[%d] = %.6f, bottom[%d] = %.6f\n",
               i, prelu_data[i], i, X_data[i]);
      }
    }
  }

  // memory deallocation
  {
    printf("free\n");
    free(X_data);
    free(relu_data);
    free(prelu_data);
  }
  #ifdef GPU
  {
    printf("gpu free\n");
    CUDA_CHECK(cudaFree(X.data));
    CUDA_CHECK(cudaFree(Y_relu.data));
    CUDA_CHECK(cudaFree(Y_prelu.data));
  }
  #endif

  return 0;
}
#endif // endifdef TEST
