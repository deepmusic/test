#include "layer.h"
#include <math.h>
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
// --------------------------------------------------------------------------

// compute max2d[n][d] = max_c(data3d[n][c][d])
#ifdef GPU
__global__
void channel_max_gpu(const real* const data3d, real* const max2d,
                     const int N, const int C, const int D)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N * D) {
    const int n = index / D;
    const int d = index % D;

    real maxval = -FLT_MAX;
    for (int c = 0; c < C; ++c) {
      maxval = MAX(maxval,  data3d[(n * C + c) * D + d]);
    }
    max2d[index] = maxval;
  }
}
#else
void channel_max_cpu(const real* const data3d, real* const max2d,
                     const int N, const int C, const int D)
{
  for (int n = 0; n < N; ++n) {
    for (int d = 0; d < D; ++d) {
      real maxval = -FLT_MAX;
      for (int c = 0; c < C; ++c) {
        maxval = MAX(maxval,  data3d[(n * C + c) * D + d]);
      }
      max2d[n * D + d] = maxval;
    }
  }
}
#endif

// in-place subtraction: data3d[n][c][d] = data3d[n][c][d] - max2d[n][d]
#ifdef GPU
__global__
void subtract_max_gpu(real* const data3d, const real* const max2d,
                      const int N, const int C, const int D)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N * C * D) {
    const int n = index / C / D;
    const int d = index % D;
    data3d[index] -= max2d[n * D + d];
  }
}
#else
void subtract_max_cpu(real* const data3d, const real* const max2d,
                      const int N, const int C, const int D)
{
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        data3d[(n * C + c) * D + d] -= max2d[n * D + d];
      } // endfor d
    } // endfor c
  } // endfor n
}
#endif

// in-place element-wise exp: data3d[n][c][d] = exp(data[n][c][d])
#ifdef GPU
__global__
void exp_gpu(real* const data, const int data_size)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < data_size) {
    data[index] = exp(data[index]);
  }
}
#else
void exp_cpu(real* const data, const int data_size)
{
  for (int index = 0; index < data_size; ++index) {
    data[index] = exp(data[index]);
  }
}
#endif

// compute sum2d[n][d] = sum_c(data3d[n][c][d])
#ifdef GPU
__global__
void channel_sum_gpu(const real* const data3d, real* const sum2d,
                     const int N, const int C, const int D)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N * D) {
    const int n = index / D;
    const int d = index % D;

    real sumval = 0;
    for (int c = 0; c < C; ++c) {
      sumval += data3d[(n * C + c) * D + d];
    }
    sum2d[index] = sumval;
  }
}
#else
void channel_sum_cpu(const real* const data3d, real* const sum2d,
                     const int N, const int C, const int D)
{
  for (int n = 0; n < N; ++n) {
    for (int d = 0; d < D; ++d) {
      real sumval = 0;
      for (int c = 0; c < C; ++c) {
        sumval += data3d[(n * C + c) * D + d];
      }
      sum2d[n * D + d] = sumval;
    }
  }
}
#endif

// in-place division: data3d[n][c][d] = data3d[n][c][d] / sum2d[n][d]
#ifdef GPU
__global__
void div_sum_gpu(real* const data3d, const real* const sum2d,
                 const int N, const int C, const int D)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N * C * D) {
    const int n = index / C / D;
    const int d = index % D;
    data3d[index] /= sum2d[n * D + d];
  }
}
#else
void div_sum_cpu(real* const data3d, const real* const sum2d,
                 const int N, const int C, const int D)
{
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        data3d[(n * C + c) * D + d] /= sum2d[n * D + d];
      } // endfor d
    } // endfor c
  } // endfor n
}
#endif



// --------------------------------------------------------------------------
// layer operator code
//   softmax_forward
//   softmax_forward_inplace
// --------------------------------------------------------------------------

// channel-wise softmax transform: bottom3d (N x C x D) -> top3d (N x C x D)
//   top[n][c][d] = exp(bottom[n][c][d]) / sum_c(exp(bottom[n][c][d]))
//   temp_data: N * D array,  temporary space for channel-wise sum or max
//     e.g., temp_data[n][d] = sum_c(exp(bottom[n][c][d]))
void softmax_forward(const Tensor* const bottom3d,
                     Tensor* const top3d,
                     real* const temp_data)
{
  // copy bottom -> top, and then perform inplace operation
  const int N = bottom3d->num_items;
  const int C = bottom3d->shape[0][0];
  const int H = bottom3d->shape[0][1];
  const int W = bottom3d->shape[0][2];
  const int data_size = N * C * H * W;

  // memcpy bottom -> top
  {
  #ifdef GPU
    cudaMemcpyAsync(top3d->data, bottom3d->data, data_size * sizeof(real),
                    cudaMemcpyDeviceToDevice);
  #else
    memcpy(top3d->data, bottom3d->data, data_size * sizeof(real));
  #endif
  }

  // set top shape (= bottom shape)
  {
    top3d->ndim = bottom3d->ndim;
    top3d->num_items = bottom3d->num_items;
    for (int n = 0; n < bottom3d->num_items; ++n) {
      for (int i = 0; i < bottom3d->ndim; ++i) {
        top3d->shape[n][i] = bottom3d->shape[n][i];
      }
    }
  }

  // perform in-place operation
  softmax_inplace_forward(top3d, temp_data);
}

// channel-wise in-place softmax transform:
//   bottom[n][c][d] = exp(bottom[n][c][d]) / sum_c(exp(bottom[n][c][d]))
void softmax_inplace_forward(Tensor* const bottom3d,
                             real* const temp_data)
{
  const int N = bottom3d->num_items;
  const int C = bottom3d->shape[0][0];
  const int H = bottom3d->shape[0][1];
  const int W = bottom3d->shape[0][2];
  const int D = H * W;

  // 1. max[n][d] = max_c(bottom[n][c][d])
  {
  #ifdef GPU
    const int num_threads = N * D;
    const int threads_per_block = 512;
    const int num_blocks = DIV_THEN_CEIL(num_threads,  threads_per_block);
    channel_max_gpu<<<num_blocks, threads_per_block>>>(
        bottom3d->data,  temp_data,  N,  C,  D);
  #else
    channel_max_cpu(bottom3d->data,  temp_data,  N,  C,  D);
  #endif
  }

  // 2. sub[n][c][d] = bottom[n][c][d] - max[n][d]
  {
  #ifdef GPU
    const int num_threads = N * C * D;
    const int threads_per_block = 512;
    const int num_blocks = DIV_THEN_CEIL(num_threads,  threads_per_block);
    subtract_max_gpu<<<num_blocks, threads_per_block>>>(
        bottom3d->data,  temp_data,  N,  C,  D);
  #else
    subtract_max_cpu(bottom3d->data,  temp_data,  N,  C,  D);
  #endif
  }

  // 3. exp[n][c][d] = exp(sub[n][c][d])
  {
  #ifdef GPU
    const int num_threads = N * C * D;
    const int threads_per_block = 512;
    const int num_blocks = DIV_THEN_CEIL(num_threads,  threads_per_block);
    exp_gpu<<<num_blocks, threads_per_block>>>(
        bottom3d->data,  num_threads);
  #else
    const int data_size = N * C * D;
    exp_cpu(bottom3d->data,  data_size);
  #endif
  }

  // 4. sum[n][d] = sum_c(exp[n][c][d])
  {
  #ifdef GPU
    const int num_threads = N * D;
    const int threads_per_block = 512;
    const int num_blocks = DIV_THEN_CEIL(num_threads,  threads_per_block);
    channel_sum_gpu<<<num_blocks, threads_per_block>>>(
        bottom3d->data,  temp_data,  N,  C,  D);
  #else
    channel_sum_cpu(bottom3d->data,  temp_data,  N,  C,  D);
  #endif
  }

  // 5. top[n][c][d] = exp[n][c][d] / sum[n][d]
  {
  #ifdef GPU
    const int num_threads = N * C * D;
    const int threads_per_block = 512;
    const int num_blocks = DIV_THEN_CEIL(num_threads,  threads_per_block);
    div_sum_gpu<<<num_blocks, threads_per_block>>>(
        bottom3d->data,  temp_data,  N,  C,  D);
  #else
    div_sum_cpu(bottom3d->data,  temp_data,  N,  C,  D);
  #endif
  }
}



// --------------------------------------------------------------------------
// layer shape calculator code
// --------------------------------------------------------------------------

void softmax_shape(const Tensor* const bottom3d,
                   Tensor* const top3d,
                   int* const temp_size)
{
  // temporary space for channel-wise sum or max: N * D
  const int N = bottom3d->num_items;
  const int H = bottom3d->shape[0][1];
  const int W = bottom3d->shape[0][2];
  const int D = H * W;
  *temp_size = N * D;

  // top shape = bottom shape
  top3d->ndim = bottom3d->ndim;
  top3d->num_items = bottom3d->num_items;
  for (int n = 0; n < bottom3d->num_items; ++n) {
    for (int i = 0; i < bottom3d->ndim; ++i) {
      top3d->shape[n][i] = bottom3d->shape[n][i];
    }
  }
}
