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

typedef unsigned short ushort;
typedef unsigned int uint;
typedef float real;

typedef struct Tensor_ {
  real* data;
  ushort ndim;
  ushort shape[5];
} Tensor;

typedef struct ConvOption_ {
  ushort kernel_h;
  ushort kernel_w;
  ushort pad_h;
  ushort pad_w;
  ushort stride_h;
  ushort stride_w;
  void* handle;
} ConvOption;

#ifdef PASS
__global__ void convert_bottom_patch(const real* bottom3d_patch,
                                     real* const bottom5d_hyperpatch,
                                     const ushort kernel_h, const ushort kernel_w,
                                     const short h_min, const short h_max,
                                     const short w_min, const short w_max,
                                     const ushort W, const uint stride_hyperpatch)
{
  real* p_bottom5d = bottom5d_hyperpatch;
  for (ushort kh = blockIdx.x * blockDim.x + threadIdx.x;
       kh < kernel_h;
       kh += blockDim.x) {
    for (ushort kw = blockIdx.y * blockDim.y + threadIdx.y;
         kw < kernel_w;
         kw += blockDim.y) {
      const ushort index = kh * W + kw;
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
                               const ushort C, const ushort H, const ushort W,
                               const ushort H5, const ushort W5,
                               const ushort kernel_h, const ushort kernel_w,
                               const short pad_h, const short pad_w,
                               const ushort stride_h, const ushort stride_w)
{
  const uint top_HW = (int)H5 * W5;
  const uint top_CHW = top_HW * C;

  // thread index: (c, h5, w5) = c*H5*W5 + h5*W5 + w5
  for (uint index = blockIdx.x * blockDim.x + threadIdx.x;
       index < top_CHW;
       index += blockDim.x) {
    // parse thread index -> (c, h5, w5)
    const ushort c = index / top_HW;
    const ushort h5 = (index / W5) % H5;
    const ushort w5 = index % W5; 
    // p_bottom5d initially points to bottom5d[c][kh = 0][kw = 0][h5][w5]
    real* p_bottom5d = bottom5d + index + (c * top_HW) * (kernel_h * kernel_w - 1);

    // (h0, w0): upper-left corner location of bottom3d's kernel patch
    const short h0 = (short)h5 * stride_h - pad_h;
    const short w0 = (short)w5 * stride_w - pad_w;
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

void forward(const Tensor* bottom4d, Tensor* const top4d,
             const Tensor* weight4d, const Tensor* bias1d,
             real* const temp_data, const real* const_data,
             const ConvOption* options)
{
  // bottom shape: N x C x H x W
  const ushort num_batch = bottom4d->shape[0];  // N
  const ushort bottom_C = bottom4d->shape[1];  // C
  const ushort bottom_H = bottom4d->shape[2];  // H
  const ushort bottom_W = bottom4d->shape[3];  // W

  // weight shape: C' x C x kernel_h x kernel_w
  const ushort top_C = weight4d->shape[0];  // C'
  const ushort kernel_h = weight4d->shape[2];
  const ushort kernel_w = weight4d->shape[3];

  // padding size & stride size
  const ushort pad_h = options->pad_h;
  const ushort pad_w = options->pad_w;
  const ushort stride_h = options->stride_h;
  const ushort stride_w = options->stride_w;

  // set top shape: N x C' x H' x W'
  //   H' = 1 + (H + 2*pad_h - kernel_h) / stride_h
  //   W' = 1 + (W + 2*pad_w - kernel_w) / stride_w
  top4d->ndim = 4;
  top4d->shape[0] = num_batch;
  top4d->shape[1] = top_C;
  top4d->shape[2] = 1 + (bottom_H + 2 * pad_h - kernel_h) / stride_h;
  top4d->shape[3] = 1 + (bottom_W + 2 * pad_w - kernel_w) / stride_w;

 { // do forward-pass
  const ushort top_H = top4d->shape[2];
  const ushort top_W = top4d->shape[3];
  const uint bottom_CHW = (uint)bottom_C * bottom_H * bottom_W;
  const uint top_CHW = (uint)top_C * top_H * top_W;

  for (int n = 0; n < num_batch; ++n) {
    // locate n-th batch
    const real* p_bottom_data = bottom4d->data + n * bottom_CHW;
    real* p_top_data = top4d->data + n * top_CHW;

    // convert bottom shape: C x H x W -> (C * kernel_h * kernel_w) x (H' * W')
    const int num_threads = 64;
    const int num_blocks = (num_threads - 1 + bottom_C * top_H * top_W) / num_threads;
    convert_bottom<<<num_blocks, num_threads>>>(p_bottom_data, temp_data,
                                                bottom_C, bottom_H, bottom_W,
                                                top_H, top_W,
                                                kernel_h, kernel_w,
                                                (short)pad_h, (short)pad_w,
                                                stride_h, stride_w);
    

   {
    const uint top_HW = (uint)top_H * top_W;
    const uint weight_col = (uint)bottom_C * kernel_h * kernel_w;
    const cublasHandle_t* cublas_handle = (cublasHandle_t*)options->handle;
    const real one = 1.0, zero = 0.0;

    // dot(weight, bottom) -> C' x H' x W'
    cublasSgemm(*cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                top_HW, top_C, weight_col,
                &one, temp_data, top_HW,
                weight4d->data, weight_col,
                &zero, p_top_data, top_HW);

    // add bias
    cublasSgemm(*cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                top_HW, top_C, 1,
                &one, const_data, top_HW,
                bias1d->data, 1,
                &one, p_top_data, top_HW);
   }
  }
 } // end forward-pass
}

void backward(Tensor *top_grad, Tensor *bottom_grad, Tensor *top_layer, Tensor *bottom_layer, ConvOption *options)
{
  return;
}

uint flatten_size(Tensor *tensor)
{
  uint size = 1;
  for (int d = 0; d < tensor->ndim; ++d)
    size *= tensor->shape[d];
  return size;
}

int main(int argc, char **argv)
{
  Tensor X, Y, W, b;
  real X_data[5000], Y_data[5000], W_data[500], b_data[50], temp_data[5000];
  real* p_temp_data;
  real* p_const_data;
  cublasHandle_t cublas_handle;

  X.ndim = 4; X.shape[0] = 2; X.shape[1] = 10; X.shape[2] = 5; X.shape[3] = 5;
  W.ndim = 4; W.shape[0] = 5; W.shape[1] = 10; W.shape[2] = 3; W.shape[3] = 3;
  b.ndim = 1; b.shape[0] = 5;
  b_data[0] = 0.1; b_data[1] = -0.1; b_data[2] = 0.2; b_data[3] = -0.2; b_data[4] = 0;

  ConvOption option;
  option.kernel_h = 3;
  option.kernel_w = 3;
  option.pad_h = 1;
  option.pad_w = 1;
  option.stride_h = 2;
  option.stride_w = 2;

  printf("set device\n");
  CUDA_CHECK(cudaSetDevice(0));
  //printf("get device\n");
  //CUDA_CHECK(cudaGetDevice(0));
  printf("cublas initialization\n");
  if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
    printf("cublas creation failed\n");
  }
  option.handle = &cublas_handle;

  printf("cuda malloc\n");
  CUDA_CHECK(cudaMalloc(&X.data, 5000*sizeof(real)));
  CUDA_CHECK(cudaMalloc(&Y.data, 5000*sizeof(real)));
  CUDA_CHECK(cudaMalloc(&W.data, 500*sizeof(real)));
  CUDA_CHECK(cudaMalloc(&b.data, 50*sizeof(real)));
  CUDA_CHECK(cudaMalloc(&p_temp_data, 5000*sizeof(real)));
  CUDA_CHECK(cudaMalloc(&p_const_data, 5000*sizeof(real)));

  printf("data loading\n");
  FILE* fp;
  fp = fopen("X.txt", "r");
  uint X_size = flatten_size(&X);
  for (int i = 0; i < X_size; ++i)
    fscanf(fp, "%f", &X_data[i]);
  fclose(fp);
  fp = fopen("W.txt", "r");
  uint W_size = flatten_size(&W);
  for (int i = 0; i < W_size; ++i)
    fscanf(fp, "%f", &W_data[i]);
  fclose(fp);
  for (int i = 0; i < 5000; ++i) {
    temp_data[i] = 1;
  }

  printf("memcopy\n");
  CUDA_CHECK(cudaMemcpy(X.data, X_data, 5000*sizeof(real), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(W.data, W_data, 500*sizeof(real), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b.data, b_data, 50*sizeof(real), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(p_const_data, temp_data, 5000*sizeof(real), cudaMemcpyHostToDevice));

  printf("do forward\n");
  forward(&X, &Y, &W, &b, p_temp_data, p_const_data, &option);

  printf("memcpy\n");
  CUDA_CHECK(cudaMemcpy(Y_data, Y.data, 5000*sizeof(real), cudaMemcpyDeviceToHost));

  printf("Y (%d x %d x %d x %d)\n", Y.shape[0], Y.shape[1], Y.shape[2], Y.shape[3]);
  for (int n = 0; n < Y.shape[0]; ++n) {
    for (int c = 0; c < Y.shape[1]; ++c) {
      for (int h = 0; h < Y.shape[2]; ++h) {
        for (int w = 0; w < Y.shape[3]; ++w) {
          printf("%03.6f ", Y_data[((n * Y.shape[1] + c) * Y.shape[2] + h) * Y.shape[3] + w]);
        }
        printf("\n");
      }
      printf("\n\n");
    }
    printf("\n\n===============================\n\n");
  }

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

  return 0;
}
