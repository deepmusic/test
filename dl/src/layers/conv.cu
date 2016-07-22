#include "layers/operator.h"
#include <string.h>

// --------------------------------------------------------------------------
// kernel code
//   conv_winograd_cpu
//   convert_bottom_{gpu, cpu}
// --------------------------------------------------------------------------

#ifndef GPU
static
void conv_winograd_cpu(const real bottom3d[],
                       const real weight4d[],
                       real temp_data[],
                       real top3d[],
                       const int top_C, const int bottom_C,
                       const int H, const int W)
{
  const int H2 = DIV_THEN_CEIL(H,  2);
  const int W2 = DIV_THEN_CEIL(W,  2);
  real* const p_weight4x4 = temp_data;
  real* const p_bottom4x4 = temp_data + top_C * bottom_C * 4 * 4;
  real* const p_temp4x4 = p_bottom4x4 + bottom_C * H2 * W2 * 4 * 4;
  real d[4][4];
  real uv[16];

  {
    const int stride = top_C * bottom_C;
    for (int k = 0; k < top_C; ++k) {
      for (int c = 0; c < bottom_C; ++c) {
        const real* const g = weight4d + (k * bottom_C + c) * 3 * 3;
        real* const u = p_weight4x4 + k * bottom_C + c;
        const real g_sum = (g[0] + g[1] + g[2] +
                            g[3] + g[4] + g[5] +
                            g[6] + g[7] + g[8]) / 4;

        u[0 * stride] = g[0];
        u[1 * stride] = (g[0] + g[1] + g[2]) / 2;
        u[2 * stride] = (g[0] - g[1] + g[2]) / 2;
        u[3 * stride] = g[2];
        u[4 * stride] = (g[0] + g[3] + g[6]) / 2;
        u[5 * stride] = g_sum;
        u[6 * stride] = g_sum - (g[1] + g[4] + g[7]) / 2;
        u[7 * stride] = (g[2] + g[5] + g[8]) / 2;
        u[8 * stride] = (g[0] - g[3] + g[6]) / 2;
        u[9 * stride] = g_sum - (g[3] + g[4] + g[5]) / 2;
        u[10 * stride] = g_sum - (g[1] + g[3] + g[5] + g[7]) / 2;
        u[11 * stride] = (g[2] - g[5] + g[8]) / 2;
        u[12 * stride] = g[6];
        u[13 * stride] = (g[6] + g[7] + g[8]) / 2;
        u[14 * stride] = (g[6] - g[7] + g[8]) / 2;
        u[15 * stride] = g[8];
      } // endfor c
    } // endfor k
  }

  {
    const int stride = bottom_C * H2 * W2;
    for (int c = 0; c < bottom_C; ++c) {
    for (int h = 0; h < H; h += 2) {
    for (int w = 0; w < W; w += 2) {
      const real* const p_patch = bottom3d + (c * H + h - 1) * W + w - 1;
      //real* const v = p_bottom4x4 + (h / 2 * W2 + w / 2) * bottom_C + c;
      real* const v = p_bottom4x4 + (c * H2 + h / 2) * W2 + w / 2;

      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          const int hh = h - 1 + j;
          const int ww = w - 1 + i;
          d[j][i] = (hh >= 0 && hh < H && ww >= 0 && ww < W) ?
                    p_patch[j * W + i] : 0;
        }
      }

      v[0 * stride] = d[0][0] - d[0][2] - d[2][0] + d[2][2];
      v[1 * stride] = d[0][1] + d[0][2] - d[2][1] - d[2][2];
      v[2 * stride] = -d[0][1] + d[0][2] + d[2][1] - d[2][2];
      v[3 * stride] = d[0][1] - d[0][3] - d[2][1] + d[2][3];
      v[4 * stride] = d[1][0] - d[1][2] + d[2][0] - d[2][2];
      v[5 * stride] = d[1][1] + d[1][2] + d[2][1] + d[2][2];
      v[6 * stride] = -d[1][1] + d[1][2] - d[2][1] + d[2][2];
      v[7 * stride] = d[1][1] - d[1][3] + d[2][1] - d[2][3];
      v[8 * stride] = -d[1][0] + d[1][2] + d[2][0] - d[2][2];
      v[9 * stride] = -d[1][1] - d[1][2] + d[2][1] + d[2][2];
      v[10 * stride] = d[1][1] - d[1][2] - d[2][1] + d[2][2];
      v[11 * stride] = -d[1][1] + d[1][3] + d[2][1] - d[2][3];
      v[12 * stride] = d[1][0] - d[1][2] - d[3][0] + d[3][2];
      v[13 * stride] = d[1][1] + d[1][2] - d[3][1] - d[3][2];
      v[14 * stride] = -d[1][1] + d[1][2] + d[3][1] - d[3][2];
      v[15 * stride] = d[1][1] - d[1][3] - d[3][1] + d[3][3];
    }}} // endfor chw
  }

  {
    const int top_area = H2 * W2;
    for (int i = 0; i < 16; ++i) {
      const real* const u = p_weight4x4 + i * top_C * bottom_C;
      const real* const v = p_bottom4x4 + i * bottom_C * top_area;
      real* const uv_ = p_temp4x4 + i * top_C * top_area;
      cblas_sgemm(CblasRowMajor,
                  CblasNoTrans,  CblasNoTrans,
                  top_C,  top_area,  bottom_C,
                  1,
                  u,  bottom_C,
                  v,  top_area,
                  0,
                  uv_,  top_area);
    }
  }

  {
    const int stride = top_C * H2 * W2;
    for (int k = 0; k < top_C; ++k) {
    for (int h = 0; h < H; h += 2) {
    for (int w = 0; w < W; w += 2) {
      const real* const uv_ = p_temp4x4 + k * H2 * W2 + h / 2 * W2 + w / 2;
      real* const y = top3d + (k * H + h) * W + w;

      for (int i = 0; i < 16; ++i) {
        uv[i] =  uv_[i * stride];
      }

      y[0] = uv[0] + uv[1] + uv[2] +
             uv[4] + uv[5] + uv[6] +
             uv[8] + uv[9] + uv[10];
      if (w + 1 < W) {
        y[1] = uv[1] - uv[2] - uv[3] +
               uv[5] - uv[6] - uv[7] +
               uv[9] - uv[10] - uv[11];
      }
      if (h + 1 < H) {
        y[W] = uv[4] + uv[5] + uv[6]
               - uv[8] - uv[9] - uv[10]
               - uv[12] - uv[13] - uv[14];
        if (w + 1 < W) {
          y[W + 1] = uv[5] - uv[6] - uv[7]
                     - uv[9] + uv[10] + uv[11]
                     - uv[13] + uv[14] + uv[15];
        }
      }
    }}} // endfor khw
  }
}
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
static
void convert_bottom_gpu(const real bottom3d[],
                        real bottom5d[],
                        const int C, const int H, const int W,
                        const int H5, const int W5,
                        const int kernel_h, const int kernel_w,
                        const int pad_h, const int pad_w,
                        const int stride_h, const int stride_w)
{
  // thread index: (c, h5, w5) = c*H5*W5 + h5*W5 + w5
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int H5W5 = H5 * W5;
  if (index < C * H5W5) {
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
    //   if (h, w) is in a zero-padded region, assign 0
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        const int h = h_start + kh;
        const int w = w_start + kw;
        p_bottom5d[(kh * kernel_w + kw) * H5W5] =
          (h >= 0 && h < H && w >= 0 && w < W) ? p_bottom3d[kh * W + kw] : 0;
      }
    }
  }
}
#else
static
void convert_bottom_cpu(const real bottom3d[],
                        real bottom5d[],
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



// --------------------------------------------------------------------------
// layer-wise operator code
// --------------------------------------------------------------------------

// convolution: bottom -> top
//   G: number of groups
//   bottom: (G * C) x H x W
//   top: (G * C') x H' x W'
//   weight: G x C' x C x kernel_h x kernel_w
//   bias: (G * C') x 1
//   temp: (G * C * kernel_h * kernel_w) x (H' * W') array
//   const: 1 x (H' * W') array,  const[i] = 1 for all i
static
void conv_forward(const Tensor* const bottom3d,
                  Tensor* const top3d,
                  const Tensor* const weight5d,
                  const Tensor* const bias1d,
                  real temp_data[],
                  const real const_data[],
                  const LayerOption* const option)
{
  // weight shape: G x C' x C x kernel_h x kernel_w
  const int G = weight5d->shape[0][0];
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
  real* p_temp_data = temp_data;

  for (int n = 0; n < bottom3d->num_items; ++n) {
    // bottom shape: (G * C) x H x W
    const int bottom_H = bottom3d->shape[n][1];  // H
    const int bottom_W = bottom3d->shape[n][2];  // W

    // set top shape: (G * C') x H' x W'
    //   H' = 1 + (H + 2*pad_h - kernel_h) / stride_h
    //   W' = 1 + (W + 2*pad_w - kernel_w) / stride_w
    const int top_H = 1 + (bottom_H + 2 * pad_h - kernel_h) / stride_h;
    const int top_W = 1 + (bottom_W + 2 * pad_w - kernel_w) / stride_w;
    top3d->shape[n][0] = G * top_C;
    top3d->shape[n][1] = top_H;
    top3d->shape[n][2] = top_W;

  #ifndef GPU
    if (top_C >= 48 &&
        kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1)
    {
      conv_winograd_cpu(
          p_bottom_item,  weight5d->data,  temp_data,  p_top_item,
          top_C, bottom_C, bottom_H, bottom_W);
      #ifdef DEBUG
      printf("%s -> %s: Winograd conv\n", bottom3d->name, top3d->name);
      #endif
    }
    else {
  #endif

    // convert bottom shape
    //   (G * C) x H x W -> (G * C * kernel_h * kernel_w) x (H' * W')
    if (kernel_h != 1 || kernel_w != 1 ||
        bottom_H != top_H || bottom_W != top_W)
    {
    #ifdef GPU
      // one thread computes "kernel_h * kernel_w" entries in top
      const int num_threads = G * bottom_C * top_H * top_W;
      const int threads_per_block = 512;
      const int num_blocks = DIV_THEN_CEIL(num_threads,  threads_per_block);
      convert_bottom_gpu<<<num_blocks, threads_per_block>>>(
          p_bottom_item,  p_temp_data,
          G * bottom_C,  bottom_H,  bottom_W,
          top_H,  top_W,
          kernel_h,  kernel_w,  pad_h,  pad_w,  stride_h,  stride_w);
    #else
      convert_bottom_cpu(
          p_bottom_item,  p_temp_data,
          G * bottom_C,  bottom_H,  bottom_W,
          top_H,  top_W,
          kernel_h,  kernel_w,  pad_h,  pad_w,  stride_h,  stride_w);
    #endif
    }
    else {
      // if 1x1 convolution, skip convert_bottom
      p_temp_data = (real*)p_bottom_item;
    }

    // compute top[g] = dot(weight[g], bottom[g])
    //   weight[g]: C' x (C * kernel_h * kernel_w)
    //   bottom[g]: (C * kernel_h * kernel_w) x (H' * W')
    //   top[g]: C' x H' x W'
    for (int g = 0; g < G; ++g) {
      const int kernel_size = bottom_C * kernel_h * kernel_w;
      const int top_area = top_H * top_W;
      const real* const p_temp_g = p_temp_data +
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
                  1,
                  p_weight_g,  kernel_size,
                  p_temp_g,  top_area,
                  0,
                  p_top_g,  top_area);
    #endif
    } // endfor g

  #ifndef GPU
    }
  #endif

    // compute top[i][j] = top[i][j] + bias[i]
    //   top: (G * C') x (H' * W')
    //   bias: (G * C') x 1
    if (option->bias) {
      const int top_channels = G * top_C;
      const int top_area = top_H * top_W;

      // the computation is equivalent to...
      //   top = top + dot(bias, constant)
      //   constant: 1 x (H' * W'),  constant[i] = 1 for all i
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
      cblas_sger(CblasRowMajor,
                 top_channels,  top_area,
                 1,
                 bias1d->data,  1,
                 const_data,  1,
                 p_top_item,  top_area);
    #endif
    }

    // locate next item
    {
      const int bottom_size = G * bottom_C * bottom_H * bottom_W;
      const int top_size = G * top_C * top_H * top_W;
      p_bottom_item += bottom_size;
      p_top_item += top_size;
    }
  } // endfor batch

  top3d->ndim = 3;
  top3d->num_items = bottom3d->num_items;
}



// --------------------------------------------------------------------------
// output & parameter shape calculator code
// --------------------------------------------------------------------------

static
void conv_shape(const Tensor* const bottom3d,
                Tensor* const top3d,
                Tensor* const weight5d,
                Tensor* const bias1d,
                long int* const p_temp_space,
                long int* const p_const_space,
                const LayerOption* const option)
{
  const int G = option->group;
  const int top_C = option->num_output / option->group;  // C'
  const int bottom_C = bottom3d->shape[0][0] / option->group;  // C
  const int kernel_h = option->kernel_h;
  const int kernel_w = option->kernel_w;
  const int pad_h = option->pad_h;
  const int pad_w = option->pad_w;
  const int stride_h = option->stride_h;
  const int stride_w = option->stride_w;

  // calculate shape for each item in the batch
  int total_size = 0, max_top_area = 0;
  for (int n = 0; n < bottom3d->num_items; ++n) {
    // bottom shape: (G * C) x H x W
    const int bottom_H = bottom3d->shape[n][1];  // H
    const int bottom_W = bottom3d->shape[n][2];  // W

    // top shape: (G * C') x H' x W'
    //   H' = 1 + (H + 2*pad_h - kernel_h) / stride_h
    //   W' = 1 + (W + 2*pad_w - kernel_w) / stride_w
    const int top_H = 1 + (bottom_H + 2 * pad_h - kernel_h) / stride_h;
    const int top_W = 1 + (bottom_W + 2 * pad_w - kernel_w) / stride_w;
    const int top_area = top_H * top_W;
    top3d->shape[n][0] = G * top_C;
    top3d->shape[n][1] = top_H;
    top3d->shape[n][2] = top_W;

    // start position for n-th item in top3d->data
    top3d->start[n] = total_size;
    total_size += G * top_C * top_H * top_W;

    // max(H' * W') in the batch
    max_top_area = MAX(max_top_area,  top_area);
  }
  top3d->ndim = 3;
  top3d->num_items = bottom3d->num_items;

  // weight shape: G x C' x C x kernel_h x kernel_w
  weight5d->num_items = 1;
  weight5d->ndim = 5;
  weight5d->shape[0][0] = G;
  weight5d->shape[0][1] = top_C;
  weight5d->shape[0][2] = bottom_C;
  weight5d->shape[0][3] = kernel_h;
  weight5d->shape[0][4] = kernel_w;
  weight5d->start[0] = 0;

  // bias shape: (G * C') x 1
  if (option->bias) {
    bias1d->num_items = 1;
    bias1d->ndim = 1;
    bias1d->shape[0][0] = G * top_C;
    bias1d->start[0] = 0;
  }
  else if (bias1d) {
    bias1d->num_items = 0;
    bias1d->ndim = 0;
    bias1d->shape[0][0] = 0;
    bias1d->start[0] = 0;
  }

  // temporary data size: G * C * kernel_h * kernel_w * max(H' * W')
  //                      + additional space for Winograd convolution
  *p_temp_space = sizeof(real) * (
      G * bottom_C * kernel_h * kernel_w * max_top_area
      + G * top_C * max_top_area * 4
      + G * top_C * bottom_C * 4 * 4);

  // constant data size: max(H' * W')
  *p_const_space = max_top_area * sizeof(real);
}



// --------------------------------------------------------------------------
// functions for layer instance
// --------------------------------------------------------------------------

void forward_conv_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;
  Tensor* const p_bias = (layer->option.bias) ? get_param(layer, 1) : NULL;

  conv_forward(get_bottom(layer, 0), get_top(layer, 0),
               get_param(layer, 0), p_bias,
               net->temp_data, net->const_data, &layer->option);
}

void shape_conv_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;
  Tensor* const p_bias = (layer->option.bias) ? get_param(layer, 1) : NULL;
  long int temp_space, const_space;

  conv_shape(get_bottom(layer, 0), get_top(layer, 0),
             get_param(layer, 0), p_bias,
             &temp_space, &const_space, &layer->option);

  update_temp_space(net, temp_space);
  update_const_space(net, const_space);
}

void init_conv_layer(void* const net_, void* const layer_)
{
  return;
}

void free_conv_layer(void* const net_, void* const layer_)
{
  return;
}
