#include "net.h"
#include <float.h>
#include <string.h>

// channel-wise softmax transform: bottom3d (N x C x D) -> top3d (N x C x D)
//   top[n][c][d] = exp(bottom[n][c][d]) / sum_c(exp(bottom[n][c][d]))
//   1. to avoid that exp(...) goes to infinity, subtract max
//     exp(x_c) / sum_c(exp(x_c))
//     =  ( exp(x_c) / exp(x_max) )  /  sum_c( exp(x_c) / exp(x_max) )
//     =  exp(x_c - x_max) / sum_c(exp(x_c - x_max))
//   2. thus, the transform cosists of 5 steps:
//     a. max[n][d] = max_c(bottom[n][c][d])
//     b. sub[n][c][d] = bottom[n][c][d] - max[n][d]
//     c. exp[n][c][d] = exp(sub[n][c][d])
//     d. sum[n][d] = sum_c(exp[n][c][d])
//     e. top[n][c][d] = exp[n][c][d] / sum[n][d]

// --------------------------------------------------------------------------
// kernel code
//   channel_max_{gpu, cpu}
//   subtract_max_{gpu, cpu}
//   exp_{gpu, cpu}
//   channel_sum_{gpu, cpu}
//   div_sum_{gpu, cpu}
//   softmax_inplace
// --------------------------------------------------------------------------

// compute max2d[n][d] = max_c(data3d[n][c][d])
#ifdef GPU
__global__
static
void channel_max_gpu(const real data3d[], real max2d[],
                     const int N, const int C, const int D)
{
  const long int index = blockIdx.x * blockDim.x + threadIdx.x;
  const long int max_size = N * D;
  if (index < max_size) {
    const int n = index / D;
    const int d = index % D;

    real maxval = -FLT_MAX;
    for (int c = 0; c < C; ++c) {
      const long int data_index = (n * C + c) * D + d;
      maxval = MAX(maxval,  data3d[data_index]);
    }
    max2d[index] = maxval;
  }
}
#else
static
void channel_max_cpu(const real data3d[], real max2d[],
                     const int N, const int C, const int D)
{
  for (int n = 0; n < N; ++n) {
    for (int d = 0; d < D; ++d) {
      const long int max_index = n * D + d;
      real maxval = -FLT_MAX;
      for (int c = 0; c < C; ++c) {
        const long int data_index = (n * C + c) * D + d;
        maxval = MAX(maxval,  data3d[data_index]);
      }
      max2d[max_index] = maxval;
    }
  }
}
#endif

// in-place subtraction: data3d[n][c][d] = data3d[n][c][d] - max2d[n][d]
#ifdef GPU
__global__
static
void subtract_max_gpu(real data3d[], const real max2d[],
                      const int N, const int C, const int D)
{
  const long int index = blockIdx.x * blockDim.x + threadIdx.x;
  const long int data_size = N * C * D;
  if (index < data_size) {
    const int n = index / C / D;
    const int d = index % D;
    const long int max_index = n * D + d;
    data3d[index] -= max2d[max_index];
  }
}
#else
static
void subtract_max_cpu(real data3d[], const real max2d[],
                      const int N, const int C, const int D)
{
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        const long int data_index = (n * C + c) * D + d;
        const long int max_index = n * D + d;
        data3d[data_index] -= max2d[max_index];
      } // endfor d
    } // endfor c
  } // endfor n
}
#endif

// in-place element-wise exp: data3d[n][c][d] = exp(data[n][c][d])
#ifdef GPU
__global__
static
void exp_gpu(real data[], const long int data_size)
{
  const long int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < data_size) {
    data[index] = exp(data[index]);
  }
}
#else
static
void exp_cpu(real data[], const long int data_size)
{
  for (long int index = 0; index < data_size; ++index) {
    data[index] = exp(data[index]);
  }
}
#endif

// compute sum2d[n][d] = sum_c(data3d[n][c][d])
#ifdef GPU
__global__
static
void channel_sum_gpu(const real data3d[], real sum2d[],
                     const int N, const int C, const int D)
{
  const long int index = blockIdx.x * blockDim.x + threadIdx.x;
  const long int sum_size = N * D;
  if (index < sum_size) {
    const int n = index / D;
    const int d = index % D;

    real sumval = 0;
    for (int c = 0; c < C; ++c) {
      const long int data_index = (n * C + c) * D + d;
      sumval += data3d[data_index];
    }
    sum2d[index] = sumval;
  }
}
#else
static
void channel_sum_cpu(const real data3d[], real sum2d[],
                     const int N, const int C, const int D)
{
  for (int n = 0; n < N; ++n) {
    for (int d = 0; d < D; ++d) {
      const long int sum_index = n * D + d;
      real sumval = 0;
      for (int c = 0; c < C; ++c) {
        const long int data_index = (n * C + c) * D + d;
        sumval += data3d[data_index];
      }
      sum2d[sum_index] = sumval;
    }
  }
}
#endif

// in-place division: data3d[n][c][d] = data3d[n][c][d] / sum2d[n][d]
#ifdef GPU
__global__
static
void div_sum_gpu(real data3d[], const real sum2d[],
                 const int N, const int C, const int D)
{
  const long int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N * C * D) {
    const int n = index / C / D;
    const int d = index % D;
    const long int sum_index = n * D + d;
    data3d[index] /= sum2d[sum_index];
  }
}
#else
static
void div_sum_cpu(real data3d[], const real sum2d[],
                 const int N, const int C, const int D)
{
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        const long int data_index = (n * C + c) * D + d;
        const long int sum_index = n * D + d;
        data3d[data_index] /= sum2d[sum_index];
      } // endfor d
    } // endfor c
  } // endfor n
}
#endif

// channel-wise in-place softmax transform
//   bottom[n][c][d] = exp(bottom[n][c][d]) / sum_c(exp(bottom[n][c][d]))
//   bottom3d: N x C x D array
//   temp_data: N * D array,  temporary space for channel-wise sum or max
//     e.g., temp_data[n][d] = sum_c(exp(bottom3d[n][c][d]))
static
void softmax_inplace(real bottom3d[], real temp_data[],
                     const int N, const int C, const int D)
{
  // 1. max[n][d] = max_c(bottom[n][c][d])
  {
  #ifdef GPU
    const long int num_threads = N * D;
    const int threads_per_block = 512;
    const int num_blocks = DIV_THEN_CEIL(num_threads,  threads_per_block);
    channel_max_gpu<<<num_blocks, threads_per_block>>>(
        bottom3d,  temp_data,  N,  C,  D);
  #else
    channel_max_cpu(bottom3d,  temp_data,  N,  C,  D);
  #endif
  }

  // 2. sub[n][c][d] = bottom[n][c][d] - max[n][d]
  {
  #ifdef GPU
    const long int num_threads = N * C * D;
    const int threads_per_block = 512;
    const int num_blocks = DIV_THEN_CEIL(num_threads,  threads_per_block);
    subtract_max_gpu<<<num_blocks, threads_per_block>>>(
        bottom3d,  temp_data,  N,  C,  D);
  #else
    subtract_max_cpu(bottom3d,  temp_data,  N,  C,  D);
  #endif
  }

  // 3. exp[n][c][d] = exp(sub[n][c][d])
  {
  #ifdef GPU
    const long int num_threads = N * C * D;
    const int threads_per_block = 512;
    const int num_blocks = DIV_THEN_CEIL(num_threads,  threads_per_block);
    exp_gpu<<<num_blocks, threads_per_block>>>(
        bottom3d,  num_threads);
  #else
    const long int data_size = N * C * D;
    exp_cpu(bottom3d,  data_size);
  #endif
  }

  // 4. sum[n][d] = sum_c(exp[n][c][d])
  {
  #ifdef GPU
    const long int num_threads = N * D;
    const int threads_per_block = 512;
    const int num_blocks = DIV_THEN_CEIL(num_threads,  threads_per_block);
    channel_sum_gpu<<<num_blocks, threads_per_block>>>(
        bottom3d,  temp_data,  N,  C,  D);
  #else
    channel_sum_cpu(bottom3d,  temp_data,  N,  C,  D);
  #endif
  }

  // 5. top[n][c][d] = exp[n][c][d] / sum[n][d]
  {
  #ifdef GPU
    const long int num_threads = N * C * D;
    const int threads_per_block = 512;
    const int num_blocks = DIV_THEN_CEIL(num_threads,  threads_per_block);
    div_sum_gpu<<<num_blocks, threads_per_block>>>(
        bottom3d,  temp_data,  N,  C,  D);
  #else
    div_sum_cpu(bottom3d,  temp_data,  N,  C,  D);
  #endif
  }
}



// --------------------------------------------------------------------------
// layer operator code
// --------------------------------------------------------------------------

// channel-wise softmax transform: bottom3d (N x C x D) -> top3d (N x C x D)
//   top[n][c][d] = exp(bottom[n][c][d]) / sum_c(exp(bottom[n][c][d]))
//   option->channel_axis: axis index to be considered as "channel"
//     e.g., option->channel_axis = 0 if bottom = C x H x W tensor
//     N = product of shape[0, ..., option->channel_axis-1]
//     C = shape[option->channel_axis]
//     D = product of shape[option->channel_axis+1, ..., ndim-1]
//   temp_data: N * D array,  temporary space for channel-wise sum or max
//     e.g., temp_data[n][d] = sum_c(exp(bottom[n][c][d]))
static
void softmax_forward(const Tensor* const bottom,
                     Tensor* const top,
                     real temp_data[],
                     const LayerOption* const option)
{
  // copy bottom -> top, and then perform inplace operation
  if (bottom->data != top->data) {
    const long int data_size = get_data_size(bottom);

    #ifdef GPU
    cudaMemcpyAsync(top->data, bottom->data, data_size * sizeof(real),
                    cudaMemcpyDeviceToDevice);
    #else
    memcpy(top->data, bottom->data, data_size * sizeof(real));
    #endif
  }

  // perform in-place softmax operation
  for (int n = 0; n < bottom->num_items; ++n) {
    real* const p_top_item = top->data + bottom->start[n];
    const int C = bottom->shape[n][option->channel_axis];
    int N = 1, D = 1;
    for (int i = 0; i < option->channel_axis; ++i) {
      N *= bottom->shape[n][i];
    }
    for (int i = option->channel_axis + 1; i < bottom->ndim; ++i) {
      D *= bottom->shape[n][i];
    }

    softmax_inplace(p_top_item, temp_data, N, C, D);
  }
}



// --------------------------------------------------------------------------
// layer shape calculator code
// --------------------------------------------------------------------------

static
void softmax_shape(const Tensor* const bottom,
                   Tensor* const top,
                   long int* const p_temp_space,
                   const LayerOption* const option)
{
  // top shape = bottom shape
  if (bottom != top) {
    top->ndim = bottom->ndim;
    top->num_items = bottom->num_items;
    for (int n = 0; n < bottom->num_items; ++n) {
      for (int i = 0; i < bottom->ndim; ++i) {
        top->shape[n][i] = bottom->shape[n][i];
      }
    }
    for (int n = 0; n < bottom->num_items; ++n) {
      top->start[n] = bottom->start[n];
    }
  }

  // temporary space for channel-wise sum or max: N * D
  {
    int ND_max = 0;
    for (int n = 0; n < bottom->num_items; ++n) {
      int N = 1, D = 1;
      for (int i = 0; i < option->channel_axis; ++i) {
        N *= bottom->shape[n][i];
      }
      for (int i = option->channel_axis + 1; i < bottom->ndim; ++i) {
        D *= bottom->shape[n][i];
      }
      ND_max = MAX(ND_max,  N * D);
    }
    *p_temp_space = ND_max * sizeof(real);
  }
}



// --------------------------------------------------------------------------
// API code
// --------------------------------------------------------------------------

void forward_softmax_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;

  softmax_forward(get_bottom(layer, 0), get_top(layer, 0),
                  net->temp_data, &layer->option);
}

void shape_softmax_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;
  long int temp_space;

  softmax_shape(get_bottom(layer, 0), get_top(layer, 0),
                &temp_space, &layer->option);

  update_temp_space(net, temp_space);
}
