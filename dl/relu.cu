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
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < data_size) {
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
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < data_size) {
    top[index] = bottom[index] * ((bottom[index] > 0) * 1 +
                                  (bottom[index] < 0) * negative_slope);
  }
}
#else
void prelu_cpu(const real* const bottom, real* const top,
               const int data_size, const real negative_slope)
{
  for (int index = 0; index < data_size; ++index) {
    top[index] = bottom[index] * ((bottom[index] > 0) * 1 +
                                  (bottom[index] < 0) * negative_slope);
  }
}
#endif

// in-place ReLU transform
#ifdef GPU
__global__
void relu_inplace_gpu(real* const bottom, const int data_size)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < data_size) {
    bottom[index] *= (bottom[index] > 0);
  }
}
#else
void relu_inplace_cpu(real* const bottom, const int data_size)
{
  for (int index = 0; index < data_size; ++index) {
    bottom[index] *= (bottom[index] > 0);
  }
}
#endif

// in-place soft ReLU transform
#ifdef GPU
__global__
void prelu_inplace_gpu(real* const bottom, const int data_size,
                       const real negative_slope)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < data_size) {
    bottom[index] *= ((bottom[index] > 0) * 1 +
                      (bottom[index] < 0) * negative_slope);
  }
}
#else
void prelu_inplace_cpu(real* const bottom, const int data_size,
                       const real negative_slope)
{
  for (int index = 0; index < data_size; ++index) {
    bottom[index] *= ((bottom[index] > 0) * 1 +
                      (bottom[index] < 0) * negative_slope);
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
    const int num_blocks = DIV_THEN_CEIL(data_size,  threads_per_block);
    if (option->negative_slope == 0) {
      relu_gpu<<<num_blocks, threads_per_block>>>(
          bottom->data,  top->data,  data_size);
    }
    else {
      prelu_gpu<<<num_blocks, threads_per_block>>>(
          bottom->data,  top->data,  data_size,  option->negative_slope);
    }
  }
  #else
  {
    if (option->negative_slope == 0) {
      relu_cpu(
          bottom->data,  top->data,  data_size);
    }
    else {
      prelu_cpu(
          bottom->data,  top->data,  data_size,  option->negative_slope);
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
    const int num_blocks = DIV_THEN_CEIL(data_size,  threads_per_block);
    if (option->negative_slope == 0) {
      relu_inplace_gpu<<<num_blocks, threads_per_block>>>(
          bottom->data,  data_size);
    }
    else {
      prelu_inplace_gpu<<<num_blocks, threads_per_block>>>(
          bottom->data,  data_size,  option->negative_slope);
    }
  }
  #else
  {
    if (option->negative_slope == 0) {
      relu_inplace_cpu(
          bottom->data,  data_size);
    }
    else {
      prelu_inplace_cpu(
          bottom->data,  data_size,  option->negative_slope);
    }
  }
  #endif
}



// --------------------------------------------------------------------------
// layer shape calculator code
// --------------------------------------------------------------------------

void relu_shape(const Tensor* const bottom,
                Tensor* const top)
{
  // top shape = bottom shape
  top->ndim = bottom->ndim;
  top->num_items = bottom->num_items;
  for (int n = 0; n < bottom->num_items; ++n) {
    for (int i = 0; i < bottom->ndim; ++i) {
      top->shape[n][i] = bottom->shape[n][i];
    }
  }
}



// --------------------------------------------------------------------------
// test code
// --------------------------------------------------------------------------

#ifdef TEST
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  // variable declaration & memory allocation
  Tensor X, Y_relu, Y_prelu;
  real *X_data = NULL, *relu_data = NULL, *prelu_data = NULL;
  ReluOption option;

  // set option
  {
    option.negative_slope = 0;
  }

  // load data
  {
    int ndim;
    int shape[g_max_ndim];
    int total_size;

    X_data = load_data("../data/temp/conv_top0.bin", &ndim, shape);
    X.num_items = shape[0];
    X.ndim = ndim - 1;
    total_size = 0;
    for (int n = 0; n < X.num_items; ++n) {
      int size_n = 1;
      for (int i = 0; i < X.ndim; ++i) {
        X.shape[n][i] = shape[i + 1];
        size_n *= shape[i + 1];
      }
      X.start[n] = total_size;
      total_size += size_n;
    }

    relu_shape(&X, &Y_relu);
    relu_shape(&X, &Y_prelu);

    relu_data = (real*)malloc(flatten_size(&Y_relu) * sizeof(real));
    prelu_data = (real*)malloc(flatten_size(&Y_prelu) * sizeof(real));
  }
 
  // CUDA initialization
  #ifdef GPU
  {
    printf("set device\n");
    cudaSetDevice(0);
  }
  #endif

  // bind loaded data to corresponding tensors
  #ifdef GPU
  {
    const int X_size = flatten_size(&X);
    const int relu_size = flatten_size(&Y_relu);
    const int prelu_size = flatten_size(&Y_prelu);

    printf("gpu malloc\n");
    cudaMalloc(&X.data, X_size * sizeof(real));
    cudaMalloc(&Y_relu.data, relu_size * sizeof(real));
    cudaMalloc(&Y_prelu.data, prelu_size * sizeof(real));

    printf("memcpy: cpu -> gpu\n");
    cudaMemcpyAsync(X.data, X_data, X_size * sizeof(real),
                    cudaMemcpyHostToDevice);
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
    const int relu_size = flatten_size(&Y_relu);
    const int prelu_size = flatten_size(&Y_prelu);

    printf("memcpy: cpu <- gpu (relu)\n");
    cudaMemcpyAsync(relu_data, Y_relu.data, relu_size * sizeof(real),
                    cudaMemcpyDeviceToHost);

    printf("memcpy: cpu <- gpu (prelu)\n");
    cudaMemcpyAsync(prelu_data, Y_prelu.data, prelu_size * sizeof(real),
                    cudaMemcpyDeviceToHost);
  }
  #endif

  // verify results
  {
    const int relu_size = flatten_size(&Y_relu);
    const int prelu_size = flatten_size(&Y_prelu);

    printf("verification (relu)\n");

    for (int i = 0; i < relu_size; ++i) {
      if (relu_data[i] != X_data[i]
          && (relu_data[i] != 0 || X_data[i] > 0)) {
        printf("top[%d] = %.6f, bottom[%d] = %.6f\n",
               i, relu_data[i], i, X_data[i]);
      }
    }

    printf("verification (prelu)\n");

    for (int i = 0; i < prelu_size; ++i) {
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
    cudaFree(X.data);
    cudaFree(Y_relu.data);
    cudaFree(Y_prelu.data);
  }
  #endif

  return 0;
}
#endif // endifdef TEST
