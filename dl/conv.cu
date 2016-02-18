#include <cblas.h>
#include <string.h>
#include <stdio.h>

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

typedef float real;

#define g_max_num_items 128
#define g_max_ndim 4

typedef struct Tensor_ {
  real* data;
  int num_items;
  int ndim;
  int shape[g_max_num_items][g_max_ndim];
} Tensor;

typedef struct ConvOption_ {
  int kernel_h;
  int kernel_w;
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  void* handle;
} ConvOption;

#ifdef PASS
__global__ void convert_bottom_patch(const real* bottom3d_patch,
                                     real* const bottom5d_hyperpatch,
                                     const int kernel_h, const int kernel_w,
                                     const int h_min, const int h_max,
                                     const int w_min, const int w_max,
                                     const int W, const int stride_hyperpatch)
{
  real* p_bottom5d = bottom5d_hyperpatch;
  for (int kh = blockIdx.x * blockDim.x + threadIdx.x;
       kh < kernel_h;
       kh += blockDim.x) {
    for (int kw = blockIdx.y * blockDim.y + threadIdx.y;
         kw < kernel_w;
         kw += blockDim.y) {
      const int index = kh * W + kw;
      if (kh >= h_min && kh < h_max && kw >= w_min && kw < w_max) {
        p_bottom5d[index * stride_hyperpatch] = bottom3d_patch[index];
      }
      else {
        p_bottom5d[index * stride_hyperpatch] = 0;
      }
    }
  }
}
#endif

// convert bottom3d (C x H x W)
//         -> bottom5d (C x kernel_h x kernel_w x H5 x W5)
//   given (c, h5, w5), for kh: [0, ..., kernel_h) and kw: [0, ..., kernel_w),
//     bottom5d[c][kh][kw][h5][w5] = bottom3d[c][h][w]
//       h = (-pad_h + stride_h * h5) + kh
//       w = (-pad_w + stride_w * w5) + kw
//       if !(0 <= h < H) or !(0 <= w < W), assign 0
__global__ void convert_bottom(const real* bottom3d, real* const bottom5d,
                               const int C, const int H, const int W,
                               const int H5, const int W5,
                               const int kernel_h, const int kernel_w,
                               const int pad_h, const int pad_w,
                               const int stride_h, const int stride_w)
{
  const int top_HW = H5 * W5;
  const int top_CHW = top_HW * C;

  // thread index: (c, h5, w5) = c*H5*W5 + h5*W5 + w5
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < top_CHW;
       index += blockDim.x) {
    // parse thread index -> (c, h5, w5)
    const int c = index / top_HW;
    const int h5 = (index / W5) % H5;
    const int w5 = index % W5; 
    // p_bottom5d initially points to bottom5d[c][kh = 0][kw = 0][h5][w5]
    real* p_bottom5d = bottom5d + index + (c * top_HW) * (kernel_h * kernel_w - 1);

    // (h0, w0): upper-left corner location of bottom3d's kernel patch
    const int h0 = h5 * stride_h - pad_h;
    const int w0 = w5 * stride_w - pad_w;
    const real* p_bottom3d = bottom3d + (c * H + h0) * W + w0;

#ifdef PASS
    dim3 num_threads(3, 3);
    dim3 num_blocks((kernel_h + 3 - 1)/3, (kernel_w + 3 - 1)/3);
    convert_bottom_patch<<<num_blocks, num_threads>>>(p_bottom3d, p_bottom5d,
                                                      kernel_h, kernel_w,
                                                      -h0, H-h0, -w0, W-w0,
                                                      W, top_HW);
#else
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        if (h0 + kh >= 0 && h0 + kh < H && w0 + kw >= 0 && w0 + kw < W) {
          // bottom5d[c][kh][kw][h5][w5] = bottom3d[c][h0 + kh][w0 + kw]
          p_bottom5d[(kh * kernel_w + kw) * top_HW] = p_bottom3d[kh * W + kw];
        }
        else {
          // if [h0 + kh][w0 + kw] is in a zero-padded region, assign 0
          p_bottom5d[(kh * kernel_w + kw) * top_HW] = 0;
        }
      }
    }
#endif
  }
}

void forward(const Tensor* bottom3d, Tensor* const top3d,
             const Tensor* weight4d, const Tensor* bias1d,
             real* const temp_data, const real* const_data,
             const ConvOption* options)
{
  // weight shape: C' x C x kernel_h x kernel_w
  const int top_C = weight4d->shape[0][0];  // C'
  const int bottom_C = weight4d->shape[0][1];  // C
  const int kernel_h = weight4d->shape[0][2];
  const int kernel_w = weight4d->shape[0][3];

  // padding size & stride size
  const int pad_h = options->pad_h;
  const int pad_w = options->pad_w;
  const int stride_h = options->stride_h;
  const int stride_w = options->stride_w;

  // do forward-pass for each item in the batch
  const real* p_bottom_data = bottom3d->data;
  real* p_top_data = top3d->data;
  const int num_items = bottom3d->num_items;
  for (int n = 0; n < num_items; ++n) {
    // bottom shape: C x H x W
    const int bottom_H = bottom3d->shape[n][1];  // H
    const int bottom_W = bottom3d->shape[n][2];  // W

    // set top shape: C' x H' x W'
    //   H' = 1 + (H + 2*pad_h - kernel_h) / stride_h
    //   W' = 1 + (W + 2*pad_w - kernel_w) / stride_w
    const int top_H = 1 + (bottom_H + 2 * pad_h - kernel_h) / stride_h;
    const int top_W = 1 + (bottom_W + 2 * pad_w - kernel_w) / stride_w;
    top3d->ndim = 3;
    top3d->num_items = num_items;
    top3d->shape[n][0] = top_C;
    top3d->shape[n][1] = top_H;
    top3d->shape[n][2] = top_W;

   { // convert bottom shape: C x H x W -> (C * kernel_h * kernel_w) x (H' * W')
    const int num_threads = 1024;
    const int num_blocks = (num_threads - 1 + bottom_C * top_H * top_W) / num_threads;
    convert_bottom<<<num_blocks, num_threads>>>(p_bottom_data, temp_data,
                                                bottom_C, bottom_H, bottom_W,
                                                top_H, top_W,
                                                kernel_h, kernel_w,
                                                pad_h, pad_w,
                                                stride_h, stride_w);
   } // end convert bottom shape

   { // do matrix computation
    const cublasHandle_t* cublas_handle = (cublasHandle_t*)options->handle;
    const real one = 1.0, zero = 0.0;

    // top = dot(weight, bottom)
    //   weight: C' x (C * kernel_h * kernel_w)
    //   bottom: (C * kernel_h * kernel_w) x (H' * W')
    //   top: C' x H' x W'
    cublasSgemm(*cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                top_H * top_W, top_C, bottom_C * kernel_h * kernel_w,
                &one, temp_data, top_H * top_W,
                weight4d->data, bottom_C * kernel_h * kernel_w,
                &zero, p_top_data, top_H * top_W);

    // top = top + bias
    cublasSgemm(*cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                top_H * top_W, top_C, 1,
                &one, const_data, top_H * top_W,
                bias1d->data, 1,
                &one, p_top_data, top_H * top_W);
   } // end matrix computation

    // locate next data
    p_bottom_data += bottom_C * bottom_H * bottom_W;
    p_top_data += top_C * top_H * top_W;
  } // endfor batch
}

void backward(Tensor *top_grad, Tensor *bottom_grad, Tensor *top_layer, Tensor *bottom_layer, ConvOption *options)
{
  return;
}

int flatten_size(const Tensor* tensor)
{
  int size = 0;
  for (int n = 0; n < tensor->num_items; ++n) {
    int size_n = 1;
    for (int d = 0; d < tensor->ndim; ++d)
      size_n *= tensor->shape[n][d];
    size += size_n;
  }
  return size;
}

int main(int argc, char **argv)
{
  Tensor X, Y, W, b;
  real X_data[5000], Y_data[5000], W_data[500], b_data[50], temp_data[5000];
  ConvOption option;
  real* p_temp_data;
  real* p_const_data;
  cublasHandle_t cublas_handle;
 {
  X.ndim = 3; X.num_items = 2;
  for (int i = 0; i < X.num_items; ++i) {
    X.shape[i][0] = 10;
    X.shape[i][1] = 5;
    X.shape[i][2] = 5;
  }
  W.ndim = 4; W.num_items = 1; W.shape[0][0] = 5; W.shape[0][1] = 10; W.shape[0][2] = 3; W.shape[0][3] = 3;
  b.ndim = 1; b.num_items = 1; b.shape[0][0] = 5;
  X.data = &X_data[0];
  Y.data = &Y_data[0];
  W.data = &W_data[0];
  b.data = &b_data[0]; b_data[0] = 0.1; b_data[1] = -0.1; b_data[2] = 0.2; b_data[3] = -0.2; b_data[4] = 0;
  option.kernel_h = 3;
  option.kernel_w = 3;
  option.pad_h = 1;
  option.pad_w = 1;
  option.stride_h = 2;
  option.stride_w = 2;
 }
 {
  printf("set device\n");
  CUDA_CHECK(cudaSetDevice(0));
  //printf("get device\n");
  //CUDA_CHECK(cudaGetDevice(0));
  printf("cublas initialization\n");
  if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
    printf("cublas creation failed\n");
  }
  option.handle = &cublas_handle;
 }
 {
  printf("cuda malloc\n");
  CUDA_CHECK(cudaMalloc(&X.data, 5000*sizeof(real)));
  CUDA_CHECK(cudaMalloc(&Y.data, 5000*sizeof(real)));
  CUDA_CHECK(cudaMalloc(&W.data, 500*sizeof(real)));
  CUDA_CHECK(cudaMalloc(&b.data, 50*sizeof(real)));
  CUDA_CHECK(cudaMalloc(&p_temp_data, 5000*sizeof(real)));
  CUDA_CHECK(cudaMalloc(&p_const_data, 5000*sizeof(real)));
 }
 {
  printf("data loading\n");
  FILE* fp;
  fp = fopen("X.txt", "r");
  int X_size = flatten_size(&X);
  for (int i = 0; i < X_size; ++i)
    fscanf(fp, "%f", &X_data[i]);
  fclose(fp);
  fp = fopen("W.txt", "r");
  int W_size = flatten_size(&W);
  for (int i = 0; i < W_size; ++i)
    fscanf(fp, "%f", &W_data[i]);
  fclose(fp);
  for (int i = 0; i < 5000; ++i) {
    temp_data[i] = 1;
  }
 }
 {
  printf("memcopy\n");
  CUDA_CHECK(cudaMemcpy(X.data, X_data, 5000*sizeof(real), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(W.data, W_data, 500*sizeof(real), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b.data, b_data, 50*sizeof(real), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(p_const_data, temp_data, 5000*sizeof(real), cudaMemcpyHostToDevice));
 }
 {
  real* p_Y_data = &Y_data[0];

  printf("do forward\n");
  forward(&X, &Y, &W, &b, p_temp_data, p_const_data, &option);

  printf("memcpy\n");
  CUDA_CHECK(cudaMemcpy(Y_data, Y.data, 5000*sizeof(real), cudaMemcpyDeviceToHost));

  for (int n = 0; n < Y.num_items; ++n) {
    printf("Y[%d] (%d x %d x %d)\n", n, Y.shape[n][0], Y.shape[n][1], Y.shape[n][2]);
    for (int c = 0; c < Y.shape[n][0]; ++c) {
      for (int h = 0; h < Y.shape[n][1]; ++h) {
        for (int w = 0; w < Y.shape[n][2]; ++w) {
          printf("%03.6f ", p_Y_data[(c * Y.shape[n][1] + h) * Y.shape[n][2] + w]);
        }
        printf("\n");
      }
      printf("\n\n");
    }
    p_Y_data += Y.shape[n][0] * Y.shape[n][1] * Y.shape[n][2];
    printf("\n\n===============================\n\n");
  }
 }
 {
  printf("cuda free\n");
  CUDA_CHECK(cudaFree(X.data));
  CUDA_CHECK(cudaFree(Y.data));
  CUDA_CHECK(cudaFree(W.data));
  CUDA_CHECK(cudaFree(b.data));
  CUDA_CHECK(cudaFree(p_temp_data));
  CUDA_CHECK(cudaFree(p_const_data));
  printf("cublas finalization\n");
  if (cublasDestroy(cublas_handle) != CUBLAS_STATUS_SUCCESS) {
    printf("cublas destruction failed\n");
  }
 }
  return 0;
}
