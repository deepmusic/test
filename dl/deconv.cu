#include "layer.h"

// --------------------------------------------------------------------------
// kernel code
//   convert_top_{gpu, cpu}
// --------------------------------------------------------------------------

// convert top5d (C x kernel_h x kernel_w x H5 x W5)
//         -> top3d (C x H x W)
//   TODO: detailed description
#ifdef GPU
__global__
void convert_top_gpu(const real* const top5d,
                     real* const top3d,
                     const int C, const int H, const int W,
                     const int H5, const int W5,
                     const int kernel_h, const int kernel_w,
                     const int pad_h, const int pad_w,
                     const int stride_h, const int stride_w)
{
  // thread index: (c, h, w) = c*H*W + h*W + w
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < C * H * W) {
    // parse thread index -> (c, h, w)
    const int c = index / (H * W);
    const int h = (index / W) % H + pad_h;
    const int w = index % W + pad_w;

    // range of summation
    // top3d[c][h][w] = sum_{h5,w5} top5d[]
    //   0 <= h5 <= 0
    //   0 <= w5 <= 0
    //   TODO: optimization & description
    const int h5_start = (h >= kernel_h) * ((h - kernel_h) / stride_h + 1);
    const int h5_end = MIN(h / stride_h + 1,  H5);
    const int w5_start = (w >= kernel_w) * ((w - kernel_w) / stride_w + 1);
    const int w5_end = MIN(w / stride_w + 1,  W5);
    const real* p_top5d = top5d +
                  (c * kernel_h * kernel_w + h * kernel_w + w) * H5 * W5;
    const int h5_coef = (1 - stride_h * kernel_w * H5) * W5;
    const int w5_coef = 1 - stride_w * H5 * W5;

    // top3d[c][h][w] = sum_{h5,w5} top5d[]
    real val = 0;
    for (int h5 = h5_start; h5 < h5_end; ++h5) {
      for (int w5 = w5_start; w5 < w5_end; ++w5) {
        val += p_top5d[h5 * h5_coef + w5 * w5_coef];
      }
    }
    top3d[index] = val;
  }
}
#else
void convert_top_cpu(const real* const top5d,
                     real* const top3d,
                     const int C, const int H, const int W,
                     const int H5, const int W5,
                     const int kernel_h, const int kernel_w,
                     const int pad_h, const int pad_w,
                     const int stride_h, const int stride_w)
{
  // thread index: (c, h, w) = c*H*W + h*W + w
  for (int index = 0; index < C * H * W; ++index) {
    // parse thread index -> (c, h, w)
    const int c = index / (H * W);
    const int h = (index / W) % H + pad_h;
    const int w = index % W + pad_w;

    // range of summation
    // top3d[c][h][w] = sum_{h5,w5} top5d[]
    //   0 <= h5 <= 0
    //   0 <= w5 <= 0
    //   TODO: optimization & description
    const int h5_start = (h >= kernel_h) * ((h - kernel_h) / stride_h + 1);
    const int h5_end = MIN(h / stride_h + 1,  H5);
    const int w5_start = (w >= kernel_w) * ((w - kernel_w) / stride_w + 1);
    const int w5_end = MIN(w / stride_w + 1,  W5);
    const real* p_top5d = top5d +
                  (c * kernel_h * kernel_w + h * kernel_w + w) * H5 * W5;
    const int h5_coef = (1 - stride_h * kernel_w * H5) * W5;
    const int w5_coef = 1 - stride_w * H5 * W5;

    // top3d[c][h][w] = sum_{h5,w5} top5d[]
    real val = 0;
    for (int h5 = h5_start; h5 < h5_end; ++h5) {
      for (int w5 = w5_start; w5 < w5_end; ++w5) {
        val += p_top5d[h5 * h5_coef + w5 * w5_coef];
      }
    }
    top3d[index] = val;
  }
}
#endif



// --------------------------------------------------------------------------
// layer operator code
//   deconv_forward
// --------------------------------------------------------------------------

// deconvolution: bottom -> top
//   G: number of groups
//   bottom: (G * C') x H' x W'
//   top: (G * C) x H x W
//   weight: G x C' x C x kernel_h x kernel_w
//   bias: (G * C) x 1
//   temp: (G * C * kernel_h * kernel_w) x (H' * W') array
//   const: 1 x (H * W) array,  const[i] = 1 for all i
void deconv_forward(const Tensor* const bottom3d,
                    Tensor* const top3d,
                    const Tensor* const weight5d,
                    const Tensor* const bias1d,
                    real* const temp_data,
                    const real* const const_data,
                    const LayerOption* const option)
{
  // weight shape: G x C' x C x kernel_h x kernel_w
  const int num_groups = weight5d->shape[0][0]; // G
  const int bottom_C = weight5d->shape[0][1];  // C'
  const int top_C = weight5d->shape[0][2];  // C
  const int kernel_h = weight5d->shape[0][3];
  const int kernel_w = weight5d->shape[0][4];

  // padding size & stride size
  const int pad_h = option->pad_h;
  const int pad_w = option->pad_w;
  const int stride_h = option->stride_h;
  const int stride_w = option->stride_w;

  #ifdef GPU
  const real* * pp_bottom = (const real* *)malloc(num_groups * sizeof(real*));
  const real* * pp_weight = (const real* *)malloc(num_groups * sizeof(real*));
  real** pp_temp = (real**)malloc(num_groups * sizeof(real*));
  const real* * pp_bottom_dev;
  const real* * pp_weight_dev;
  real** pp_temp_dev;
  cudaMalloc(&pp_bottom_dev, num_groups * sizeof(real*));
  cudaMalloc(&pp_weight_dev, num_groups * sizeof(real*));
  cudaMalloc(&pp_temp_dev, num_groups * sizeof(real*));
  #endif

  // do forward-pass for each item in the batch
  const real* p_bottom_item = bottom3d->data;
  real* p_top_item = top3d->data;
  for (int n = 0; n < bottom3d->num_items; ++n) {
    // bottom shape: (G * C') x H' x W'
    const int bottom_H = bottom3d->shape[n][1];  // H'
    const int bottom_W = bottom3d->shape[n][2];  // W'

    // set top shape: (G * C) x H x W
    //   H' = 1 + (H + 2 * pad_h - kernel_h) / stride_h
    //   -> H = stride_h * (H' - 1) - 2 * pad_h + kernel_h
    const int top_H = stride_h * (bottom_H - 1) - 2 * pad_h + kernel_h;
    const int top_W = stride_w * (bottom_W - 1) - 2 * pad_w + kernel_w;
    top3d->shape[n][0] = num_groups * top_C;
    top3d->shape[n][1] = top_H;
    top3d->shape[n][2] = top_W;

    const int kernel_size = top_C * kernel_h * kernel_w;
    const int bottom_area = bottom_H * bottom_W;

    #ifdef GPU
    {
      // compute top[g] = dot(weight[g].transpose(), bottom[g])
      //   weight[g]: C' x (C * kernel_h * kernel_w)
      //   bottom[g]: C' x (H' * W')
      //   top[g]: (C * kernel_h * kernel_w) x (H' * W')
      for (int g = 0; g < num_groups; ++g) {
        pp_bottom[g] = p_bottom_item + g * bottom_C * bottom_area;
        pp_weight[g] = weight5d->data + g * bottom_C * kernel_size;
        pp_temp[g] = temp_data + g * kernel_size * bottom_area;
      }
      cudaMemcpyAsync(pp_bottom_dev, pp_bottom, num_groups * sizeof(real*),
                      cudaMemcpyHostToDevice);
      cudaMemcpyAsync(pp_weight_dev, pp_weight, num_groups * sizeof(real*),
                      cudaMemcpyHostToDevice);
      cudaMemcpyAsync(pp_temp_dev, pp_temp, num_groups * sizeof(real*),
                      cudaMemcpyHostToDevice);

      // compute Z = alpha * dot(X.transpose(), Y) + beta * Z
      //   X (= weight): p x m,  Y (= bottom): p x n,  Z (= top): m x n
      //   X, Y, Z: row-major order (e.g., Z[i][j] = Z[i * n + j])
      // input arguments:
      //   cublas handle,
      //   do_transpose_Y (= false),  do_transpose_X (= true),
      //   n (= H' * W'),  m (= C * kernel_h * kernel_w),  p (= C'),
      //   &alpha (= 1),
      //   &Y,  number of columns in Y (= n),
      //   &X,  number of columns in X (= m),
      //   &beta (= 0),
      //   &Z,  number of columns in Z (= n)
      const real one = 1.0f, zero = 0.0f;
      cublasSgemmBatched(*((cublasHandle_t*)option->handle),
                  CUBLAS_OP_N,  CUBLAS_OP_T,
                  bottom_area,  kernel_size,  bottom_C,
                  &one,
                  pp_bottom_dev,  bottom_area,
                  pp_weight_dev,  kernel_size,
                  &zero,
                  pp_temp_dev,  bottom_area, num_groups);
    }
    #else
    {
      // compute top[g] = dot(weight[g].transpose(), bottom[g])
      //   weight[g]: C' x (C * kernel_h * kernel_w)
      //   bottom[g]: C' x (H' * W')
      //   top[g]: (C * kernel_h * kernel_w) x (H' * W')
      for (int g = 0; g < num_groups; ++g) {
        const real* const p_bottom_g = p_bottom_item +
                                       g * bottom_C * bottom_area;
        const real* const p_weight_g = weight5d->data +
                                       g * bottom_C * kernel_size;
        real* const p_temp_g = temp_data + g * kernel_size * bottom_area;

        // compute Z = alpha * dot(X.transpose(), Y) + beta * Z
        //   X (= weight): p x m,  Y (= bottom): p x n,  Z (= top): m x n
        //   X, Y, Z: row-major order (e.g., Z[i][j] = Z[i * n + j])
        // input arguments:
        //   is_row_major_order (= true),
        //   do_transpose_X (= true),  do_transpose_Y (= false),
        //   m (= C * kernel_h * kernel_w),  n (= H' * W'),  p (= C'),
        //   alpha (= 1),
        //   &X,  number of columns in X (= m),
        //   &Y,  number of columns in Y (= n),
        //   beta (= 0),
        //   &Z,  number of columns in Z (= n)
        cblas_sgemm(CblasRowMajor,
                    CblasTrans,  CblasNoTrans,
                    kernel_size,  bottom_area,  bottom_C,
                    1,
                    p_weight_g,  kernel_size,
                    p_bottom_g,  bottom_area,
                    0,
                    p_temp_g,  bottom_area);
      }
    }
    #endif

    // convert top shape
    //   (G * C * kernel_h * kernel_w) x (H' * W') -> (G * C) x (H * W)
    {
    #ifdef GPU
      // one thread computes one entry in top
      const int num_threads = num_groups * top_C * top_H * top_W;
      const int threads_per_block = 512;
      const int num_blocks = DIV_THEN_CEIL(num_threads,  threads_per_block);
      convert_top_gpu<<<num_blocks, threads_per_block>>>(
          temp_data,  p_top_item,
          num_groups * top_C,  top_H,  top_W,
          bottom_H,  bottom_W,
          kernel_h,  kernel_w,  pad_h,  pad_w,  stride_h,  stride_w);
    #else
      convert_top_cpu(
          temp_data,  p_top_item,
          num_groups * top_C,  top_H,  top_W,
          bottom_H,  bottom_W,
          kernel_h,  kernel_w,  pad_h,  pad_w,  stride_h,  stride_w);
    #endif
    }

    // compute top[i][j] = top[i][j] + bias[i]
    //   top: (G * C) x (H * W)
    //   bias: (G * C) x 1
    if (option->bias) {
      const int top_channels = num_groups * top_C;
      const int top_area = top_H * top_W;

      // the computation is equivalent to...
      //   top = top + dot(bias, constant)
      //   constant: 1 x (H * W),  constant[i] = 1 for all i
    #ifdef GPU
      // thus, input arguments:
      //   do_transpose_Y (= false),  do_transpose_X (= false),
      //   n = H * W,  m = G * C,  p = 1
      //   alpha = 1,  beta = 1
      const real one = 1.0;
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
      //   m = G * C,  n = H * W,  p = 1
      //   alpha = 1,  beta = 1
/*
      cblas_sgemm(CblasRowMajor,
                  CblasNoTrans,  CblasNoTrans,
                  top_channels,  top_area,  1,
                  1,
                  bias1d->data,  1,
                  const_data,  top_area,
                  1,
                  p_top_item,  top_area);
*/
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
      const int bottom_size = num_groups * bottom_C * bottom_H * bottom_W;
      const int top_size = num_groups * top_C * top_H * top_W;
      p_bottom_item += bottom_size;
      p_top_item += top_size;
    }
  } // endfor batch

  #ifdef GPU
  free(pp_bottom);
  free(pp_weight);
  free(pp_temp);
  cudaFree(pp_bottom_dev);
  cudaFree(pp_weight_dev);
  cudaFree(pp_temp_dev);
  #endif

  top3d->ndim = 3;
  top3d->num_items = bottom3d->num_items;
}



// --------------------------------------------------------------------------
// layer shape calculator code
// --------------------------------------------------------------------------

void deconv_shape(const Tensor* const bottom3d,
                  Tensor* const top3d,
                  Tensor* const weight5d,
                  Tensor* const bias1d,
                  int* const temp_size,
                  int* const const_size,
                  const LayerOption* const option)
{
  const int num_groups = option->num_groups; // G
  const int bottom_C = bottom3d->shape[0][0] / option->num_groups;  // C'
  const int top_C = option->out_channels / option->num_groups;  // C
  const int kernel_h = option->kernel_h;
  const int kernel_w = option->kernel_w;
  const int pad_h = option->pad_h;
  const int pad_w = option->pad_w;
  const int stride_h = option->stride_h;
  const int stride_w = option->stride_w;

  // calculate shape for each item in the batch
  int max_bottom_area = 0;
  int max_top_area = 0;
  int total_size = 0;
  for (int n = 0; n < bottom3d->num_items; ++n) {
    // bottom shape: (G * C') x H' x W'
    const int bottom_H = bottom3d->shape[n][1];  // H'
    const int bottom_W = bottom3d->shape[n][2];  // W'
    const int bottom_area = bottom_H * bottom_W;

    // top shape: (G * C) x H x W
    //   H' = 1 + (H + 2 * pad_h - kernel_h) / stride_h
    //   -> H = stride_h * (H' - 1) - 2 * pad_h + kernel_h
    const int top_H = stride_h * (bottom_H - 1) - 2 * pad_h + kernel_h;
    const int top_W = stride_w * (bottom_W - 1) - 2 * pad_w + kernel_w;
    const int top_area = top_H * top_W;
    top3d->shape[n][0] = num_groups * top_C;
    top3d->shape[n][1] = top_H;
    top3d->shape[n][2] = top_W;

    // start position for n-th item in top3d->data
    top3d->start[n] = total_size;
    total_size += num_groups * top_C * top_H * top_W;

    // max_n(H' * W') in the batch
    max_bottom_area = MAX(max_bottom_area,  bottom_area);
    // max_n(H * W) in the batch
    max_top_area = MAX(max_top_area,  top_area);
  }
  top3d->ndim = 3;
  top3d->num_items = bottom3d->num_items;

  // weight shape: G x C' x C x kernel_h x kernel_w
  weight5d->num_items = 1;
  weight5d->ndim = 5;
  weight5d->shape[0][0] = num_groups;
  weight5d->shape[0][1] = bottom_C;
  weight5d->shape[0][2] = top_C;
  weight5d->shape[0][3] = kernel_h;
  weight5d->shape[0][4] = kernel_w;
  weight5d->start[0] = 0;

  // bias shape: (G * C) x 1
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

  // temporary data size: G * C * kernel_h * kernel_w * max_n(H' * W')
  *temp_size = num_groups * top_C * kernel_h * kernel_w * max_bottom_area;

  // constant data size: max_n(H * W)
  *const_size = max_top_area;
}



// --------------------------------------------------------------------------
// API code
// --------------------------------------------------------------------------

void forward_deconv_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;

  Tensor* p_bias = (layer->option.bias) ? layer->p_params[1] : NULL;

  deconv_forward(layer->p_bottoms[0], layer->p_tops[0],
                 layer->p_params[0], p_bias,
                 net->temp_data, net->const_data, &layer->option);

  print_tensor_info(layer->name, layer->p_tops[0]);
}

void shape_deconv_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;

  int temp_size, const_size;
  Tensor* p_bias = (layer->option.bias) ? layer->p_params[1] : NULL;

  deconv_shape(layer->p_bottoms[0], layer->p_tops[0],
               layer->p_params[0], p_bias,
               &temp_size, &const_size, &layer->option);

  update_net_size(net, layer, temp_size, 0, const_size);
}



// --------------------------------------------------------------------------
// test code
// --------------------------------------------------------------------------

#ifdef TEST

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
    option.num_groups = 512;
    option.out_channels = 512;
    option.kernel_h = 4;
    option.kernel_w = 4;
    option.pad_h = 1;
    option.pad_w = 1;
    option.stride_h = 2;
    option.stride_w = 2;
    option.bias = 0;
  }

  // load data
  {
    int ndim;
    int shape[g_max_ndim];
    int total_size;

    X_data = load_data("../data/temp/deconv_bottom0.bin",
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
    deconv_shape(&X, &Y, &W, &b, &temp_size, &const_size, &option);

    Y_true_data = load_data("../data/temp/deconv_top0.bin",
                            &ndim, shape, NULL);
    Y_data = (real*)malloc(flatten_size(&Y) * sizeof(real));

    W_data = load_data("../data/temp/deconv_param0.bin",
                       &ndim, shape, NULL);

    if (option.bias) {
      b_data = load_data("../data/temp/deconv_param1.bin",
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
    const long int X_size = flatten_size(&X);
    const long int Y_size = flatten_size(&Y);
    const long int W_size = flatten_size(&W);
    const long int b_size = flatten_size(&b);

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
    deconv_forward(&X, &Y, &W, &b, p_temp_data, p_const_data, &option);
  }

  // copy GPU data to main memory
  #ifdef GPU
  {
    const long int Y_size = flatten_size(&Y);

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
