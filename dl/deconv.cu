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
static
void convert_top_gpu(const real top5d[], real top3d[],
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
    const real* const p_top5d = top5d +
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
static
void convert_top_cpu(const real top5d[], real top3d[],
                     const int C, const int H, const int W,
                     const int H5, const int W5,
                     const int kernel_h, const int kernel_w,
                     const int pad_h, const int pad_w,
                     const int stride_h, const int stride_w)
{
  const int h5_coef = (1 - stride_h * kernel_w * H5) * W5;
  const int w5_coef = 1 - stride_w * H5 * W5;
  real* p_top3d = top3d;

  for (int c = 0; c < C; ++c) {
  for (int h = pad_h; h < H + pad_h; ++h) {
    const int h5_start = (h < kernel_h) ?
                         0: ((h - kernel_h) / stride_h + 1);
    const int h5_end = MIN(h / stride_h + 1,  H5);

    for (int w = pad_w; w < W + pad_w; ++w) {
      const int w5_start = (w < kernel_w) ?
                           0 : ((w - kernel_w) / stride_w + 1);
      const int w5_end = MIN(w / stride_w + 1,  W5);
      const real* const p_top5d = top5d +
                    (c * kernel_h * kernel_w + h * kernel_w + w) * H5 * W5;

      // top3d[c][h][w] = sum_{h5,w5} top5d[]
      real val = 0;
      for (int h5 = h5_start; h5 < h5_end; ++h5) {
        for (int w5 = w5_start; w5 < w5_end; ++w5) {
          val += p_top5d[h5 * h5_coef + w5 * w5_coef];
        }
      }
      *(p_top3d++) = val;
    }
  }}
}
#endif



// --------------------------------------------------------------------------
// layer operator code
//   deconv_forward
// --------------------------------------------------------------------------

#ifdef GPU
// data structure for batched-CuBLAS operation
//   p_groups: "(3 * num_groups) x 1" array of pointers to each group
//     p_bottoms = p_groups[0, ..., num_groups - 1]
//     p_tops = p_groups[num_groups, ..., 2 * num_groups - 1]
//     p_weights = p_groups[2 * num_groups, ..., 3 * num_groups - 1]
//   p_groups_gpu: "(3 * num_groups) x 1" array in GPU memory
typedef struct DeconvAuxData_ {
  real** p_groups;
  real** p_groups_gpu;
} DeconvAuxData;

static
void malloc_deconv_aux_data(DeconvAuxData* const aux_data,
                            const int num_groups,
                            int* const p_space_cpu,
                            int* const p_space_gpu)
{
  aux_data->p_groups = (real**)malloc(3 * num_groups * sizeof(real*));
  cudaMalloc(&aux_data->p_groups_gpu, 3 * num_groups * sizeof(real*));

  *p_space_cpu = 3 * num_groups * sizeof(real*);
  *p_space_gpu = 3 * num_groups * sizeof(real*);
}

static
void free_deconv_aux_data(DeconvAuxData* const aux_data)
{
  free(aux_data->p_groups);
  cudaFree(aux_data->p_groups_gpu);

  memset(aux_data, 0, sizeof(DeconvAuxData));
}
#endif

// deconvolution: bottom -> top
//   G: number of groups
//   bottom: (G * C') x H' x W'
//   top: (G * C) x H x W
//   weight: G x C' x C x kernel_h x kernel_w
//   bias: (G * C) x 1
//   temp: (G * C * kernel_h * kernel_w) x (H' * W') array
//   const: 1 x (H * W) array,  const[i] = 1 for all i
static
void deconv_forward(const Tensor* const bottom3d,
                    Tensor* const top3d,
                    const Tensor* const weight5d,
                    const Tensor* const bias1d,
                    real temp_data[],
                    const real const_data[],
                    void* const aux_data,
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
  real** const p_groups = ((DeconvAuxData*)aux_data)->p_groups;
  real** const p_groups_gpu = ((DeconvAuxData*)aux_data)->p_groups_gpu;

  const real** const p_bottoms = (const real**)&p_groups[0];
  const real** const p_weights = (const real**)&p_groups[num_groups];
  real** const p_temps = &p_groups[2 * num_groups];

  const real** const p_bottoms_gpu = (const real**)&p_groups_gpu[0];
  const real** const p_weights_gpu = (const real**)&p_groups_gpu[num_groups];
  real** const p_temps_gpu = &p_groups_gpu[2 * num_groups];
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
      const real one = 1.0f, zero = 0.0f;

      // compute top[g] = dot(weight[g].transpose(), bottom[g])
      //   weight[g]: C' x (C * kernel_h * kernel_w)
      //   bottom[g]: C' x (H' * W')
      //   top[g]: (C * kernel_h * kernel_w) x (H' * W')
      for (int g = 0; g < num_groups; ++g) {
        p_bottoms[g] = p_bottom_item + g * bottom_C * bottom_area;
        p_weights[g] = weight5d->data + g * bottom_C * kernel_size;
        p_temps[g] = temp_data + g * kernel_size * bottom_area;
      }
      cudaMemcpyAsync(p_groups_gpu, p_groups, 3 * num_groups * sizeof(real*),
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
      cublasSgemmBatched(*((cublasHandle_t*)option->handle),
                  CUBLAS_OP_N,  CUBLAS_OP_T,
                  bottom_area,  kernel_size,  bottom_C,
                  &one,
                  p_bottoms_gpu,  bottom_area,
                  p_weights_gpu,  kernel_size,
                  &zero,
                  p_temps_gpu,  bottom_area, num_groups);
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

  top3d->ndim = 3;
  top3d->num_items = bottom3d->num_items;
}



// --------------------------------------------------------------------------
// layer shape calculator code
// --------------------------------------------------------------------------

static
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
  Tensor* const p_bias = (layer->option.bias) ? layer->p_params[1] : NULL;

  deconv_forward(layer->p_bottoms[0], layer->p_tops[0],
                 layer->p_params[0], p_bias,
                 net->temp_data, net->const_data,
                 layer->aux_data,  &layer->option);
}

void shape_deconv_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;
  Tensor* const p_bias = (layer->option.bias) ? layer->p_params[1] : NULL;
  int temp_size, const_size;

  deconv_shape(layer->p_bottoms[0], layer->p_tops[0],
               layer->p_params[0], p_bias,
               &temp_size, &const_size, &layer->option);

  update_net_size(net, layer, temp_size, 0, const_size);
}

void malloc_deconv_layer(void* const net_, void* const layer_)
{
  #ifdef GPU
  {
    Net* const net = (Net*)net_;
    Layer* const layer = (Layer*)layer_;
    int space_cpu = 0, space_gpu = 0;

    layer->aux_data = (void*)malloc(sizeof(DeconvAuxData));

    malloc_deconv_aux_data((DeconvAuxData*)layer->aux_data,
                           layer->option.num_groups,
                           &space_cpu, &space_gpu);

    net->space_cpu += space_cpu + sizeof(DeconvAuxData);
    net->space += space_gpu;
  }
  #endif
}

void free_deconv_layer(void* const net_, void* const layer_)
{
  #ifdef GPU
  {
    Layer* const layer = (Layer*)layer_;

    free_deconv_aux_data((DeconvAuxData*)layer->aux_data);
    free(layer->aux_data);
  }
  #endif
}
