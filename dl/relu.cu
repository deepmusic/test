#include "layer.h"
#include "cuda_settings.h"

// relu transform bottom -> top
//   top[i] = 0 if bottom[i] <= 0
__global__ void relu(const real* const bottom, real* const top,
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

// soft relu transform bottom -> top
//   top[i] = slope * bottom[i] if bottom[i] <= 0
__global__ void prelu(const real* const bottom, real* const top,
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

// in-place relu transform
__global__ void relu_inplace(real* const bottom, const int data_size)
{
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < data_size;
       index += blockDim.x) {
    if (bottom[index] < 0) {
      bottom[index] = 0;
    }
  }
}

// in-place soft relu transform
__global__ void prelu_inplace(real* const bottom, const int data_size,
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

// relu transform: bottom -> top
//   data size: whole size (N * C * H * W or something)
void forward(const Tensor* const bottom, Tensor* const top,
             const ReluOption* const option)
{
  const int data_size = flatten_size(bottom);
  const int threads_per_block = 512;
  const int num_blocks = DIV_THEN_CEIL(data_size, threads_per_block);

  if (option->negative_slope == 0) {
    relu<<<num_blocks, threads_per_block>>>(
        bottom->data, top->data, data_size);
  }
  else {
    prelu<<<num_blocks, threads_per_block>>>(
        bottom->data, top->data, data_size, option->negative_slope);
  }

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
void forward_inplace(Tensor* const bottom,
             const ReluOption* const option)
{
  const int data_size = flatten_size(bottom);
  const int threads_per_block = 512;
  const int num_blocks = DIV_THEN_CEIL(data_size, threads_per_block);

  if (option->negative_slope == 0) {
    relu_inplace<<<num_blocks, threads_per_block>>>(
        bottom->data, data_size);
  }
  else {
    prelu_inplace<<<num_blocks, threads_per_block>>>(
        bottom->data, data_size, option->negative_slope);
  }
}

#define DATA_SIZE 128*72*92

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

    fp = fopen("../data/temp/conv_top0.txt", "r");
    for (int i = 0; i < X_size; ++i)
      fscanf(fp, "%f", &X_data[i]);
    fclose(fp);
  }

  {
    printf("set device\n");
    CUDA_CHECK(cudaSetDevice(1));
  }

  {
    int X_size = flatten_size(&X);
    int Y_size = flatten_size(&X);

    printf("cuda malloc\n");
    CUDA_CHECK(cudaMalloc(&X.data, X_size*sizeof(real)));
    CUDA_CHECK(cudaMalloc(&Y.data, Y_size*sizeof(real)));

    printf("memcopy\n");
    CUDA_CHECK(cudaMemcpy(X.data, X_data, X_size*sizeof(real), cudaMemcpyHostToDevice));
  }

  {
    printf("do forward\n");
    forward(&X, &Y, &option);
  }

  {
    int Y_size = flatten_size(&Y);
    printf("memcpy\n");
    CUDA_CHECK(cudaMemcpy(Y_data, Y.data, Y_size*sizeof(real), cudaMemcpyDeviceToHost));

    for (int i = 0; i < Y_size; ++i) {
      if (Y_data[i] != X_data[i] && (Y_data[i] != 0 || X_data[i] > 0)) {
        printf("top[%d] = %.6f, bottom[%d] = %.6f\n", i, Y_data[i], i, X_data[i]);
      }
    }
  }

  {
    printf("cuda free\n");
    CUDA_CHECK(cudaFree(X.data));
    CUDA_CHECK(cudaFree(Y.data));

    printf("free\n");
    free(X_data);
    free(Y_data);
  }

  return 0;
}
