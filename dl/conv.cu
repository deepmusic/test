#include "layer.h"

#ifdef GPU
#include "cuda_settings.h"
#else
#include <cblas.h>
#endif

// convert bottom3d (C x H x W)
//         -> bottom5d (C x kernel_h x kernel_w x H5 x W5)
//   given (c, h5, w5),
//     bottom5d[c][kh][kw][h5][w5] = bottom3d[c][h][w]
//       h = (-pad_h + stride_h * h5) + kh,  kh = { 0, 1, ..., kernel_h - 1 }
//       w = (-pad_w + stride_w * w5) + kw,  kw = { 0, 1, ..., kernel_w - 1 }
//       if !(0 <= h < H) or !(0 <= w < W), assign 0
#ifdef GPU
__global__
void convert_bottom_gpu(const real* const bottom3d,
                        real* const bottom5d,
                        const int C, const int H, const int W,
                        const int H5, const int W5,
                        const int kernel_h, const int kernel_w,
                        const int pad_h, const int pad_w,
                        const int stride_h, const int stride_w)
{
  const int H5W5 = H5 * W5;

  // thread index: (c, h5, w5) = c*H5*W5 + h5*W5 + w5
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < C * H5W5;
       index += blockDim.x) {
    // parse thread index -> (c, h5, w5)
    const int c = index / H5W5;
    const int h5 = (index / W5) % H5;
    const int w5 = index % W5; 
    // p_bottom5d initially points to bottom5d[c][kh = 0][kw = 0][h5][w5]
    real* p_bottom5d = bottom5d + index +
                       (c * H5W5) * (kernel_h * kernel_w - 1);

    // (h_start, w_start): upper-left corner of bottom3d's kernel patch
    const int h_start = h5 * stride_h - pad_h;
    const int w_start = w5 * stride_w - pad_w;
    const real* p_bottom3d = bottom3d + (c * H + h_start) * W + w_start;

    // bottom5d[c][kh][kw][h5][w5] = bottom3d[c][h][w]
    //   h = h_start + kh,  kh = {0, 1, ..., kernel_h - 1}
    //   w = w_start + kw,  kw = {0, 1, ..., kernel_w - 1}
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        if (h_start + kh >= 0 && h_start + kh < H &&
            w_start + kw >= 0 && w_start + kw < W) {
          p_bottom5d[(kh * kernel_w + kw) * H5W5] = p_bottom3d[kh * W + kw];
        }
        else {
          // if (h, w) is in a zero-padded region, assign 0
          p_bottom5d[(kh * kernel_w + kw) * H5W5] = 0;
        }
      }
    }
  }
}
#else
void convert_bottom_cpu(const real* const bottom3d,
                        real* const bottom5d,
                        const int C, const int H, const int W,
                        const int H5, const int W5,
                        const int kernel_h, const int kernel_w,
                        const int pad_h, const int pad_w,
                        const int stride_h, const int stride_w)
{
  for (int c = 0; c < C; ++c) {
   for (int kh = 0; kh < kernel_h; ++kh) {
    for (int kw = 0; kw < kernel_w; ++kw) {
      // pointer to bottom5d[c][kh][kw][h5 = 0][w5 = 0]
      real* const p_bottom5d = bottom5d +
                    ((c * kernel_h + kh) * kernel_w + kw) * H5 * W5;
      int h = -pad_h + kh;
      int h5 = 0;

      // for h < 0 (zero-padded region): bottom5d[c][kh][kw][h5][:] = 0
      for (; h < 0; h += stride_h, ++h5) {
        for (int w5 = 0; w5 < W5; ++w5) {
          p_bottom5d[h5 * W5 + w5] = 0;
        }
      }

      // for 0 <= h < H (data region)
      for (; h < H && h5 < H5; h += stride_h, ++h5) {
        // pointer to bottom3d[c][h][w = 0]
        int w = -pad_w + kw;
        int w5 = 0;

        // for w < 0 (zero-padded region): bottom5d[c][kh][kw][h5][w5] = 0
        for (; w < 0; w += stride_w, ++w5) {
          p_bottom5d[h5 * W5 + w5] = 0;
        }

        // for 0 <= w < W (data region):
        //   bottom5d[c][kh][kw][h5][w5] = bottom3d[c][h][w]
        for (; w < W && w5 < W5; w += stride_w, ++w5) {
          p_bottom5d[h5 * W5 + w5] = bottom3d[(c * H + h) * W + w];
        }

        // for w >= W (zero-padded region): bottom5d[c][kh][kw][h5][w5] = 0
        for (; w5 < W5; ++w5) {
          p_bottom5d[h5 * W5 + w5] = 0;
        }
      }

      // for h >= H (zero-padded region): bottom5d[c][kh][kw][h5][:] = 0
      for (; h5 < H5; ++h5) {
        for (int w5 = 0; w5 < W5; ++w5) {
          p_bottom5d[h5 * W5 + w5] = 0;
        }
      }
    } // endfor kw
   } // endfor kh
  } // endfor c
}
#endif

// convolution: bottom -> top
//   bottom: (G * C) x H x W
//   top: (G * C') x H' x W'
//   weight: G x C' x C x kernel_h x kernel_w
//   bias: (G * C') x 1
//   temp: G * C * kernel_h * kernel_w * H' * W'
//   const: H' * W',  const[i] = 1 for all i
//   G: number of groups
void conv_forward(const Tensor* const bottom3d,
                  Tensor* const top3d,
                  const Tensor* const weight5d,
                  const Tensor* const bias1d,
                  real* const temp_data,
                  const real* const const_data,
                  const ConvOption* const option)
{
  // weight shape: G x C' x C x kernel_h x kernel_w
  const int num_groups = weight5d->shape[0][0]; // G
  const int top_C = weight5d->shape[0][1];  // C'
  const int bottom_C = weight5d->shape[0][2];  // C
  const int kernel_h = weight5d->shape[0][3];
  const int kernel_w = weight5d->shape[0][4];

  // padding size & stride size
  const int pad_h = option->pad_h;
  const int pad_w = option->pad_w;
  const int stride_h = option->stride_h;
  const int stride_w = option->stride_w;

  // do forward-pass for each item in the batch
  const real* p_bottom_item = bottom3d->data;
  real* p_top_item = top3d->data;
  for (int n = 0; n < bottom3d->num_items; ++n) {
    // bottom shape: (G * C) x H x W
    const int bottom_H = bottom3d->shape[n][1];  // H
    const int bottom_W = bottom3d->shape[n][2];  // W

    // set top shape: (G * C') x H' x W'
    //   H' = 1 + (H + 2*pad_h - kernel_h) / stride_h
    //   W' = 1 + (W + 2*pad_w - kernel_w) / stride_w
    const int top_H = 1 + (bottom_H + 2 * pad_h - kernel_h) / stride_h;
    const int top_W = 1 + (bottom_W + 2 * pad_w - kernel_w) / stride_w;
    top3d->shape[n][0] = num_groups * top_C;
    top3d->shape[n][1] = top_H;
    top3d->shape[n][2] = top_W;

    // convert bottom shape
    //   (G * C) x H x W -> (G * C * kernel_h * kernel_w) x (H' * W')
    {
    #ifdef GPU
      // one thread computes "kernel_h * kernel_w" entries in top
      const int num_threads = num_groups * bottom_C * top_H * top_W;
      const int threads_per_block = 512;
      const int num_blocks = DIV_THEN_CEIL(num_threads, threads_per_block);
      convert_bottom_gpu<<<num_blocks, threads_per_block>>>(
          p_bottom_item,  temp_data,
          num_groups * bottom_C,  bottom_H,  bottom_W,
          top_H,  top_W,
          kernel_h,  kernel_w,  pad_h,  pad_w,  stride_h,  stride_w);
    #else
      convert_bottom_cpu(
          p_bottom_item,  temp_data,
          num_groups * bottom_C,  bottom_H,  bottom_W,
          top_H,  top_W,
          kernel_h,  kernel_w,  pad_h,  pad_w,  stride_h,  stride_w);
    #endif
    }

    // compute top[g] = dot(weight[g], bottom[g])
    //   weight[g]: C' x (C * kernel_h * kernel_w)
    //   bottom[g]: (C * kernel_h * kernel_w) x (H' * W')
    //   top[g]: C' x H' x W'
    for (int g = 0; g < num_groups; ++g) {
      const int kernel_size = bottom_C * kernel_h * kernel_w;
      const int top_area = top_H * top_W;
      const real* const p_temp_g = temp_data +
                          g * kernel_size * top_area;
      const real* const p_weight_g = weight5d->data +
                          g * top_C * kernel_size;
      real* const p_top_g = p_top_item + g * top_C * top_area;

      // compute Z = alpha * dot(X, Y) + beta * Z
      //   X (= weight): m x p,  Y (= bottom): p x n,  Z (= top): m x n
      //   X, Y, Z: row-major order (e.g., Z[i][j] = Z[i * n + j])
    #ifdef GPU
      // input arguments:
      //   cublas handle,
      //   do_transpose_Y (= false),  do_transpose_X (= false),
      //   n (= H' * W'),  m (= C'),  p (= C * kernel_h * kernel_w),
      //   &alpha (= 1),
      //   &Y,  number of columns in Y (= n),
      //   &X,  number of columns in X (= p),
      //   &beta (= 0),
      //   &Z,  number of columns in Z (= n)
      const real one = 1.0f, zero = 0.0f;
      cublasSgemm(*((cublasHandle_t*)option->handle),
                  CUBLAS_OP_N,  CUBLAS_OP_N,
                  top_area,  top_C,  kernel_size,
                  &one,
                  p_temp_g,  top_area,
                  p_weight_g,  kernel_size,
                  &zero,
                  p_top_g,  top_area);
    #else
      // input arguments:
      //   is_row_major_order (= true),
      //   do_transpose_X (= false),  do_transpose_Y (= false),
      //   m (= C'),  n (= H' * W'),  p (= C * kernel_h * kernel_w),
      //   alpha (= 1),
      //   &X,  number of columns in X (= p),
      //   &Y,  number of columns in Y (= n),
      //   beta (= 0),
      //   &Z,  number of columns in Z (= n)
      cblas_sgemm(CblasRowMajor,
                  CblasNoTrans,  CblasNoTrans,
                  top_C,  top_area,  kernel_size,
                  1.0f,
                  p_weight_g,  kernel_size,
                  p_temp_g,  top_area,
                  0.0f,
                  p_top_g,  top_area);
    #endif
    }

    // compute top[i][j] = top[i][j] + bias[i]
    //   top: (G * C') x (H' * W')
    //   bias: (G * C') x 1
    if (option->bias) {
      const int top_channels = num_groups * top_C;
      const int top_area = top_H * top_W;

      // the computation is equivalent to...
      //   top = top + dot(bias, constant)
      //   constant: 1 x (H' * W'), constant[i] = 1 for all i
    #ifdef GPU
      // thus, input arguments:
      //   do_transpose_Y (= false),  do_transpose_X (= false),
      //   n = H' * W',  m = G * C',  p = 1
      //   alpha = 1,  beta = 1
      const real one = 1.0f;
      cublasSgemm(*((cublasHandle_t*)option->handle),
                  CUBLAS_OP_N,  CUBLAS_OP_N,
                  top_area,  top_channels,  1,
                  &one,
                  const_data,  top_area,
                  bias1d->data,  1,
                  &one,
                  p_top_item,  top_area);
    #else
      // input arguments:
      //   do_transpose_X (= false),  do_transpose_Y (= false),
      //   m = G * C',  n = H' * W',  p = 1
      //   alpha = 1,  beta = 1
      cblas_sgemm(CblasRowMajor,
                  CblasNoTrans,  CblasNoTrans,
                  top_channels,  top_area,  1,
                  1.0f,
                  bias1d->data,  1,
                  const_data,  top_area,
                  1.0f,
                  p_top_item,  top_area);
    #endif
    }

    // locate next item
    {
      const int bottom_size = num_groups * bottom_C * bottom_H * bottom_W;
      const int top_size = num_groups * top_C * top_H * top_W;
      p_bottom_item += bottom_size;
      p_top_item += top_size;
    }
  } // endfor batch

  top3d->ndim = 3;
  top3d->num_items = bottom3d->num_items;
}

// test code
#ifdef TEST
#include <stdio.h>
#include <stdlib.h>

#define DATA_SIZE 384*36*46
#define WEIGHT_SIZE 384*256*3*3
#define BIAS_SIZE 384
#define CONST_SIZE 36*46

int main(int argc, char *argv[])
{
  // variable declaration & memory allocation
  Tensor X, Y, W, b;
  real* X_data = (real*)malloc(DATA_SIZE * sizeof(real));
  real* Y_data = (real*)malloc(DATA_SIZE * sizeof(real));
  real* Y_true_data = (real*)malloc(DATA_SIZE * sizeof(real));
  real* W_data = (real*)malloc(WEIGHT_SIZE * sizeof(real));
  real* b_data = (real*)malloc(BIAS_SIZE * sizeof(real));
  real* const_data = (real*)malloc(CONST_SIZE * sizeof(real));
  real* p_temp_data;
  real* p_const_data;
  ConvOption option;

  // set option
  {
    option.num_groups = 1;
    option.out_channels = 384;
    option.kernel_h = 3;
    option.kernel_w = 3;
    option.pad_h = 1;
    option.pad_w = 1;
    option.stride_h = 2;
    option.stride_w = 2;
    option.bias = 1;
  }

  // set data shapes
  {
    X.ndim = 3;
    X.num_items = 1;
    for (int i = 0; i < X.num_items; ++i) {
      X.shape[i][0] = 256;
      X.shape[i][1] = 36;
      X.shape[i][2] = 46;
    }

    Y.ndim = X.ndim;
    Y.num_items = X.num_items;
    for (int i = 0; i < Y.num_items; ++i) {
      Y.shape[i][0] = option.out_channels;
      Y.shape[i][1] = 1 + (X.shape[i][1] + 2 * option.pad_h - option.kernel_h)
                          / option.stride_h;
      Y.shape[i][2] = 1 + (X.shape[i][2] + 2 * option.pad_w - option.kernel_w)
                          / option.stride_w;
    }

    W.ndim = 5; W.num_items = 1;
    W.shape[0][0] = option.num_groups;
    W.shape[0][1] = option.out_channels / option.num_groups;
    W.shape[0][2] = X.shape[0][0] / option.num_groups;
    W.shape[0][3] = option.kernel_h;
    W.shape[0][4] = option.kernel_w;

    b.ndim = 1; b.num_items = 1;
    b.shape[0][0] = option.out_channels;
  }
 
  // load data
  {
    FILE* fp;
    int X_size = flatten_size(&X);
    int Y_size = flatten_size(&Y);
    int W_size = flatten_size(&W);
    int b_size = flatten_size(&b);

    printf("data loading\n");

    fp = fopen("../data/temp/conv_bottom0.bin", "rb");
    if ((int)fread(X_data, sizeof(real), X_size, fp) != X_size) {
      printf("Error while reading conv_bottom0\n");
    }
    fclose(fp);

    fp = fopen("../data/temp/conv_param0.bin", "rb");
    if ((int)fread(W_data, sizeof(real), W_size, fp) != W_size) {
      printf("Error while reading conv_param0\n");
    }
    fclose(fp);

    if (option.bias) {
      fp = fopen("../data/temp/conv_param1.bin", "rb");
      if ((int)fread(b_data, sizeof(real), b_size, fp) != b_size) {
        printf("Error while reading conv_param1\n");
      }
      fclose(fp);
      for (int i = 0; i < CONST_SIZE; ++i) {
        const_data[i] = 1;
      }
    }

    fp = fopen("../data/temp/conv_top0.bin", "rb");
    if ((int)fread(Y_true_data, sizeof(real), Y_size, fp) != Y_size) {
      printf("Error while reading conv_top0\n");
    }
    fclose(fp);
  }

  // CUDA initialization
  #ifdef GPU
  {
    printf("set device\n");
    CUDA_CHECK(cudaSetDevice(0));
    printf("cublas initialization\n");
    option.handle = (cublasHandle_t*)malloc(sizeof(cublasHandle_t));
    if (cublasCreate((cublasHandle_t*)option.handle)
          != CUBLAS_STATUS_SUCCESS) {
      printf("cublas creation failed\n");
    }
  }
  #endif

  // bind loaded data to corresponding tensors
  #ifdef GPU
  {
    int X_size = flatten_size(&X);
    int Y_size = flatten_size(&Y);
    int W_size = flatten_size(&W);
    int b_size = flatten_size(&b);
    int temp_size = option.kernel_h * option.kernel_w *
                    X.shape[0][0] * Y.shape[0][1] * Y.shape[0][2];

    printf("gpu malloc\n");
    CUDA_CHECK(cudaMalloc(&X.data, X_size * sizeof(real)));
    CUDA_CHECK(cudaMalloc(&Y.data, Y_size * sizeof(real)));
    CUDA_CHECK(cudaMalloc(&W.data, W_size * sizeof(real)));
    CUDA_CHECK(cudaMalloc(&b.data, b_size * sizeof(real)));
    CUDA_CHECK(cudaMalloc(&p_temp_data, temp_size * sizeof(real)));
    CUDA_CHECK(cudaMalloc(&p_const_data, CONST_SIZE * sizeof(real)));

    printf("memcpy: cpu -> gpu\n");
    CUDA_CHECK(cudaMemcpy(X.data, X_data, X_size * sizeof(real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(W.data, W_data, W_size * sizeof(real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.data, b_data, b_size * sizeof(real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p_const_data, const_data,
                          CONST_SIZE * sizeof(real),
                          cudaMemcpyHostToDevice));
  }
  #else
  {
    int temp_size = option.kernel_h * option.kernel_w *
                    X.shape[0][0] * Y.shape[0][1] * Y.shape[0][2];

    X.data = X_data;
    Y.data = Y_data;
    W.data = W_data;
    b.data = b_data;
    p_temp_data = (real*)malloc(temp_size * sizeof(real));
    p_const_data = const_data;
  }
  #endif

  // do forward operation
  {
    printf("do forward\n");
    conv_forward(&X, &Y, &W, &b, p_temp_data, p_const_data, &option);
  }

  // copy GPU data to main memory
  #ifdef GPU
  {
    int Y_size = flatten_size(&Y);
    printf("memcpy: cpu <- gpu\n");
    CUDA_CHECK(cudaMemcpy(Y_data, Y.data, Y_size * sizeof(real),
                          cudaMemcpyDeviceToHost));
  }
  #endif

  // verify results
  {
    int i = 0;
    for (int n = 0; n < Y.num_items; ++n) {
      for (int c = 0; c < Y.shape[n][0]; ++c) {
        for (int h = 0; h < Y.shape[n][1]; ++h) {
          for (int w = 0; w < Y.shape[n][2]; ++w) {
            real diff = ABS(Y_data[i] - Y_true_data[i]);
            diff /= 1e-10f + MIN(ABS(Y_data[i]), ABS(Y_true_data[i]));
          #ifdef GPU
            if (diff > 0) {
              printf("Y[%d,%d,%d,%d] = %.6f  Y_true[%d,%d,%d,%d] = %.6f\n",
                     n, c, h, w, Y_data[i], n, c, h, w, Y_true_data[i]);
            }
          #else
            if (diff > 1e-3f) {
              printf("Y[%d,%d,%d,%d] = %.6f  Y_true[%d,%d,%d,%d] = %.6f\n",
                     n, c, h, w, Y_data[i], n, c, h, w, Y_true_data[i]);
            }
          #endif
            ++i;
          } // endfor w
        } // endfor h
      } // endfor c
    } // endfor n
  }

  // memory deallocation
  {
    printf("free\n");
    free(X_data);
    free(Y_data);
    free(Y_true_data);
    free(W_data);
    free(b_data);
    free(const_data);
  }
  #ifdef GPU
  {
    printf("gpu free\n");
    CUDA_CHECK(cudaFree(X.data));
    CUDA_CHECK(cudaFree(Y.data));
    CUDA_CHECK(cudaFree(W.data));
    CUDA_CHECK(cudaFree(b.data));
    CUDA_CHECK(cudaFree(p_temp_data));
    CUDA_CHECK(cudaFree(p_const_data));

    printf("cublas finalization\n");
    if (cublasDestroy(*((cublasHandle_t*)option.handle)) != CUBLAS_STATUS_SUCCESS) {
      printf("cublas destruction failed\n");
    }
    free(option.handle);
  }
  #else
  {
    free(p_temp_data);
  }
  #endif

  return 0;
}
#endif // endifdef TEST
