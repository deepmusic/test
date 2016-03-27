#include "layer.h"

// --------------------------------------------------------------------------
// kernel code
//   convert_bottom_{gpu, cpu}
// --------------------------------------------------------------------------

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
        p_bottom5d[(kh * kernel_w + kw) * H5W5] = p_bottom3d[kh * W + kw]
            * (h >= 0) * (h < H) * (w >= 0) * (w < W);
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



// --------------------------------------------------------------------------
// layer operator code
//   conv_forward
// --------------------------------------------------------------------------

// convolution: bottom -> top
//   G: number of groups
//   bottom: (G * C) x H x W
//   top: (G * C') x H' x W'
//   weight: G x C' x C x kernel_h x kernel_w
//   bias: (G * C') x 1
//   temp: (G * C * kernel_h * kernel_w) x (H' * W') array
//   const: 1 x (H' * W') array,  const[i] = 1 for all i
void conv_forward(const Tensor* const bottom3d,
                  Tensor* const top3d,
                  const Tensor* const weight5d,
                  const Tensor* const bias1d,
                  real* const temp_data,
                  const real* const const_data,
                  const LayerOption* const option)
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
      const int num_blocks = DIV_THEN_CEIL(num_threads,  threads_per_block);
      convert_bottom_gpu<<<num_blocks, threads_per_block>>>(
          p_bottom_item,  p_temp_data,
          num_groups * bottom_C,  bottom_H,  bottom_W,
          top_H,  top_W,
          kernel_h,  kernel_w,  pad_h,  pad_w,  stride_h,  stride_w);
    #else
      convert_bottom_cpu(
          p_bottom_item,  p_temp_data,
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
      //const int temp_size =
      //    num_groups * bottom_C * kernel_h * kernel_w * top_H * top_W;
      p_bottom_item += bottom_size;
      p_top_item += top_size;
      //p_temp_data += temp_size;
    }
  } // endfor batch

  top3d->ndim = 3;
  top3d->num_items = bottom3d->num_items;
}



// --------------------------------------------------------------------------
// layer shape calculator code
// --------------------------------------------------------------------------

void conv_shape(const Tensor* const bottom3d,
                Tensor* const top3d,
                Tensor* const weight5d,
                Tensor* const bias1d,
                int* const temp_size,
                int* const const_size,
                const LayerOption* const option)
{
  const int num_groups = option->num_groups; // G
  const int top_C = option->out_channels / option->num_groups;  // C'
  const int bottom_C = bottom3d->shape[0][0] / option->num_groups;  // C
  const int kernel_h = option->kernel_h;
  const int kernel_w = option->kernel_w;
  const int pad_h = option->pad_h;
  const int pad_w = option->pad_w;
  const int stride_h = option->stride_h;
  const int stride_w = option->stride_w;

  // calculate shape for each item in the batch
  int total_size = 0, total_top_area = 0, max_top_area = 0;
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
    top3d->shape[n][0] = num_groups * top_C;
    top3d->shape[n][1] = top_H;
    top3d->shape[n][2] = top_W;

    // start position for n-th item in top3d->data
    top3d->start[n] = total_size;
    total_size += num_groups * top_C * top_H * top_W;

    // sum(H' * W') & max(H' * W') in the batch
    total_top_area += top_area;
    max_top_area = MAX(max_top_area,  top_area);
  }
  top3d->ndim = 3;
  top3d->num_items = bottom3d->num_items;

  // weight shape: G x C' x C x kernel_h x kernel_w
  weight5d->num_items = 1;
  weight5d->ndim = 5;
  weight5d->shape[0][0] = num_groups;
  weight5d->shape[0][1] = top_C;
  weight5d->shape[0][2] = bottom_C;
  weight5d->shape[0][3] = kernel_h;
  weight5d->shape[0][4] = kernel_w;
  weight5d->start[0] = 0;

  // bias shape: (G * C') x 1
  if (option->bias) {
    bias1d->num_items = 1;
    bias1d->ndim = 1;
    bias1d->shape[0][0] = num_groups * top_C;
    bias1d->start[0] = 0;
  }
  else if (bias1d) {
    bias1d->num_items = 0;
    bias1d->ndim = 0;
    bias1d->shape[0][0] = 0;
    bias1d->start[0] = 0;
  }

  // temporary data size: G * C * kernel_h * kernel_w * sum(H' * W')
  *temp_size = num_groups * bottom_C * kernel_h * kernel_w * max_top_area;

  // constant data size: max(H' * W')
  *const_size = max_top_area;
}



// --------------------------------------------------------------------------
// API code
// --------------------------------------------------------------------------

void forward_conv_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;

  Tensor* p_bias = (layer->option.bias) ? &layer->params[1] : NULL;

  conv_forward(layer->p_bottoms[0], &layer->tops[0],
               &layer->params[0], p_bias,
               net->temp_data, net->const_data, &layer->option);

  print_tensor_info(layer->name, &layer->tops[0]);
}

void shape_conv_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;

  int temp_size, const_size;
  Tensor* p_bias = (layer->option.bias) ? &layer->params[1] : NULL;

  conv_shape(layer->p_bottoms[0], &layer->tops[0],
             &layer->params[0], p_bias,
             &temp_size, &const_size, &layer->option);

  update_net_size(net, layer, temp_size, 0, const_size);
}



// --------------------------------------------------------------------------
// test code
// --------------------------------------------------------------------------

#ifdef TEST
#include <stdio.h>

int main(int argc, char* argv[])
{
  // variable declaration & memory allocation
  Tensor X, Y, W, b;
  real *X_data = NULL, *Y_data = NULL, *Y_true_data = NULL;
  real *W_data = NULL, *b_data = NULL;
  real *p_temp_data = NULL, *const_data = NULL, *p_const_data = NULL;
  LayerOption option;
  int temp_size, const_size;

  // set option
  {
    option.num_groups = 1;
    option.out_channels = 512;
    option.kernel_h = 1;
    option.kernel_w = 1;
    option.pad_h = 0;
    option.pad_w = 0;
    option.stride_h = 1;
    option.stride_w = 1;
    option.bias = 1;
  }

  // load data
  {
    int ndim;
    int shape[g_max_ndim];
    int total_size;

    X_data = load_data("../data/temp/conv_bottom0.bin",
                       &ndim, shape, NULL);
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
    conv_shape(&X, &Y, &W, &b, &temp_size, &const_size, &option);

    Y_true_data = load_data("../data/temp/conv_top0.bin",
                            &ndim, shape, NULL);
    Y_data = (real*)malloc(flatten_size(&Y) * sizeof(real));

    W_data = load_data("../data/temp/conv_param0.bin",
                       &ndim, shape, NULL);

    if (option.bias) {
      b_data = load_data("../data/temp/conv_param1.bin",
                         &ndim, shape, NULL);

      const_data = (real*)malloc(const_size * sizeof(real));
      for (int i = 0; i < const_size; ++i) {
        const_data[i] = 1;
      }
    }
  }

  // CUDA initialization
  #ifdef GPU
  {
    printf("set device\n");
    cudaSetDevice(0);
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
    const int X_size = flatten_size(&X);
    const int Y_size = flatten_size(&Y);
    const int W_size = flatten_size(&W);
    const int b_size = flatten_size(&b);

    printf("gpu malloc\n");
    cudaMalloc(&X.data, X_size * sizeof(real));
    cudaMalloc(&Y.data, Y_size * sizeof(real));
    cudaMalloc(&W.data, W_size * sizeof(real));
    cudaMalloc(&p_temp_data, temp_size * sizeof(real));
    if (option.bias) {
      cudaMalloc(&b.data, b_size * sizeof(real));
      cudaMalloc(&p_const_data, const_size * sizeof(real));
    }
    else {
      b.data = NULL;
    }

    printf("memcpy: cpu -> gpu\n");
    cudaMemcpyAsync(X.data, X_data, X_size * sizeof(real),
                    cudaMemcpyHostToDevice);
    cudaMemcpyAsync(W.data, W_data, W_size * sizeof(real),
                    cudaMemcpyHostToDevice);
    if (option.bias) {
      cudaMemcpyAsync(b.data, b_data, b_size * sizeof(real),
                      cudaMemcpyHostToDevice);
      cudaMemcpyAsync(p_const_data, const_data, const_size * sizeof(real),
                      cudaMemcpyHostToDevice);
    }
  }
  #else
  {
    X.data = X_data;
    Y.data = Y_data;
    W.data = W_data;
    p_temp_data = (real*)malloc(temp_size * sizeof(real));
    if (option.bias) {
      b.data = b_data;
      p_const_data = const_data;
    }
    else {
      b.data = NULL;
    }
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
    const int Y_size = flatten_size(&Y);

    printf("memcpy: cpu <- gpu\n");
    cudaMemcpyAsync(Y_data, Y.data, Y_size * sizeof(real),
                    cudaMemcpyDeviceToHost);
  }
  #endif

  // verify results
  {
    int i = 0;

    printf("verification\n");

    for (int n = 0; n < Y.num_items; ++n) {
      for (int c = 0; c < Y.shape[n][0]; ++c) {
        for (int h = 0; h < Y.shape[n][1]; ++h) {
          for (int w = 0; w < Y.shape[n][2]; ++w) {
            real diff = ABS(Y_data[i] - Y_true_data[i]);
            diff /= 1e-10f + MIN(ABS(Y_data[i]),  ABS(Y_true_data[i]));
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
    if (option.bias) {
      free(b_data);
      free(const_data);
    }
  }
  #ifdef GPU
  {
    printf("gpu free\n");
    cudaFree(X.data);
    cudaFree(Y.data);
    cudaFree(W.data);
    cudaFree(p_temp_data);
    if (option.bias) {
      cudaFree(b.data);
      cudaFree(p_const_data);
    }

    if (cublasDestroy(*((cublasHandle_t*)option.handle))
        != CUBLAS_STATUS_SUCCESS) {
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
