#include "layer.h"
#include <limit.h>
#include <string.h>

// --------------------------------------------------------------------------
// kernel code
//   softmax_{gpu, cpu}
//   softmax_inplace_{gpu, cpu}
// --------------------------------------------------------------------------

#ifdef GPU
__global__
void channel_max_gpu(const real* const bottom3d, real* const max2d,
                     const int N, const int C, const int D)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N * D) {
    const int n = index / D;
    const int d = index % D;

    real maxval = -FLT_MAX;
    for (int c = 0; c < C; ++c) {
      maxval = MAX(maxval,  bottom3d[(n * C + c) * D + d]);
    }
    max2d[index] = maxval;
  }
}
#else
void channel_max_cpu(const real* const bottom3d, real* const max2d,
                     const int N, const int C, const int D)
{
  for (int n = 0; n < N; ++n) {
    for (int d = 0; d < D; ++d) {
      real maxval = -FLT_MAX;
      for (int c = 0; c < C; ++c) {
        maxval = MAX(maxval,  bottom3d[(n * C + c) * D + d]);
      }
      max2d[n * D + d] = maxval;
    }
  }
}
#endif

#ifdef GPU
__global__
void channel_subtract_gpu(const real* const max2d, real* const top3d,
                          const int N, const int C, const int D)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N * C * D) {
    const int n = index / C / D;
    const int d = index % D;
    top3d[index] -= max2d[n * D + d];
  }
}
#else
void channel_subtract_cpu(const real* const max2d, real* const top3d,
                          const int N, const int C, const int D)
{
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        top3d[(n * C + c) * D + d] -= max2d[n * D + d];
      } // endfor d
    } // endfor c
  } // endfor n
}
#endif

#ifdef GPU
__global__
void exp_inplace_gpu(real* const top, const int data_size)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < data_size) {
    top[index] = exp(top[index]);
  }
}
#else
void exp_inplace_cpu(real* const top, const int data_size)
{
  for (int index = 0; index < data_size; ++index) {
    top[index] = exp(top[index]);
  }
}
#endif

#ifdef GPU
__global__
void channel_sum_gpu(const real* const bottom3d, real* const sum2d,
                     const int N, const int C, const int D)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N * D) {
    const int n = index / D;
    const int d = index % D;

    real sumval = 0;
    for (int c = 0; c < C; ++c) {
      sumval += bottom3d[(n * C + c) * D + d];
    }
    sum2d[index] = sumval;
  }
}
#else
void channel_sum_cpu(const real* const bottom3d, real* const sum2d,
                     const int N, const int C, const int D)
{
  for (int n = 0; n < N; ++n) {
    for (int d = 0; d < D; ++d) {
      real sumval = 0;
      for (int c = 0; c < C; ++c) {
        sumval += bottom3d[(n * C + c) * D + d];
      }
      sum2d[n * D + d] = sumval;
    }
  }
}
#endif

#ifdef GPU
__global__
void channel_div_gpu(const real* const sum2d, real* const top3d,
                     const int N, const int C, const int D)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N * C * D) {
    const int n = index / C / D;
    const int d = index % D;
    top3d[index] -= sum2d[n * D + d];
  }
}
#else
void channel_div_cpu(const real* const sum2d, real* const top3d,
                     const int N, const int C, const int D)
{
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        top3d[(n * C + c) * D + d] /= sum2d[n * D + d];
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
  const int C = bottom3d->shape[0];
  const int H = bottom3d->shape[1];
  const int W = bottom3d->shape[2];
  const int D = H * W;

  #ifdef GPU
  {
    const int num_threads = num_groups * bottom_C * top_H * top_W;
    const int threads_per_block = 512;
    const int num_blocks = DIV_THEN_CEIL(num_threads,  threads_per_block);
    channel_max_gpu<<<num_blocks, threads_per_block>>>(
        bottom3d->data,  temp_data,  N,  C,  D);
    
  }
  #else
    channel_max_cpu(
        bottom3d->data,  temp_data,  N,  C,  D);
  #endif
}



// --------------------------------------------------------------------------
// layer shape calculator code
// --------------------------------------------------------------------------

void softmax_shape(const Tensor* const bottom3d,
                   Tensor* const top3d)
{
  // top shape = bottom shape
  top3d->ndim = bottom3d->ndim;
  top3d->num_items = bottom3d->num_items;
  for (int n = 0; n < bottom3d->num_items; ++n) {
    for (int i = 0; i < bottom3d->ndim; ++i) {
      top3d->shape[n][i] = bottom3d->shape[n][i];
    }
  }
}
