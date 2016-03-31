#include "layer.h"
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
void channel_max_gpu(const real* const data3d, real* const max2d,
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
void channel_max_cpu(const real* const data3d, real* const max2d,
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
void subtract_max_gpu(real* const data3d, const real* const max2d,
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
void subtract_max_cpu(real* const data3d, const real* const max2d,
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
void exp_gpu(real* const data, const long int data_size)
{
  const long int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < data_size) {
    data[index] = exp(data[index]);
  }
}
#else
void exp_cpu(real* const data, const long int data_size)
{
  for (long int index = 0; index < data_size; ++index) {
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
void channel_sum_cpu(const real* const data3d, real* const sum2d,
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
void div_sum_gpu(real* const data3d, const real* const sum2d,
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
void div_sum_cpu(real* const data3d, const real* const sum2d,
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
void softmax_inplace(real* const bottom3d,
                     real* const temp_data,
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
  const long int data_size = flatten_size(bottom3d);

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
    for (int n = 0; n < bottom3d->num_items; ++n) {
      top3d->start[n] = bottom3d->start[n];
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
  // do forward-pass for each item in the batch
  if (bottom3d->ndim == 4) {
    for (int n = 0; n < bottom3d->num_items; ++n) {
      const int N = bottom3d->shape[n][0];
      const int C = bottom3d->shape[n][1];
      const int H = bottom3d->shape[n][2];
      const int W = bottom3d->shape[n][3];
      const int D = H * W;
      real* const p_bottom_item = bottom3d->data + bottom3d->start[n];
      real* const p_temp_data = temp_data + bottom3d->start[n];
      softmax_inplace(p_bottom_item, p_temp_data, N, C, D);
    } // endfor batch
  }

  else {
    for (int n = 0; n < bottom3d->num_items; ++n) {
      const int N = 1;
      const int C = bottom3d->shape[n][0];
      const int H = bottom3d->shape[n][1];
      const int W = bottom3d->shape[n][2];
      const int D = H * W;
      real* const p_bottom_item = bottom3d->data + bottom3d->start[n];
      real* const p_temp_data = temp_data + bottom3d->start[n];
      softmax_inplace(p_bottom_item, p_temp_data, N, C, D);
    } // endfor batch
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
  for (int n = 0; n < bottom3d->num_items; ++n) {
    top3d->start[n] = bottom3d->start[n];
  }
}



// --------------------------------------------------------------------------
// API code
// --------------------------------------------------------------------------

void forward_rpn_pred_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;

  // 3d tensor: C x H x W
  Tensor* const score = &layer->tops[0];

  // reshape to 3d tensor: 2 x (C / 2) x (H * W)
  score->ndim = 3;
  for (int n = 0; n < score->num_items; ++n) {
    const int C = score->shape[n][0];
    const int H = score->shape[n][1];
    const int W = score->shape[n][2];
    score->shape[n][0] = 2;
    score->shape[n][1] = C / 2;
    score->shape[n][2] = H * W;

    // backup for reshape to 4d tensor
    score->shape[n][3] = W;
  }

  // softmax transform
  softmax_inplace_forward(score, net->temp_data);

  // reshape to 4d tensor: 2 x (C / 2) x H x W
  score->ndim = 4;
  for (int n = 0; n < score->num_items; ++n) {
    score->shape[n][2] /= score->shape[n][3];
  }

  print_tensor_info(layer->name, &layer->tops[0]);
}

void shape_rpn_pred_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;

  // 3d tensor: C x H x W
  Tensor* const score = &layer->tops[0];

  // reshape to 4d tensor: 2 x (C / 2) x H x W
  score->ndim = 4;
  for (int n = 0; n < score->num_items; ++n) {
    const int C = score->shape[n][0];
    const int H = score->shape[n][1];
    const int W = score->shape[n][2];
    score->shape[n][0] = 2;
    score->shape[n][1] = C / 2;
    score->shape[n][2] = H;
    score->shape[n][3] = W;
  }
}

void forward_rpn_bbox_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;

  shape_rpn_bbox_layer(net_, layer);

  print_tensor_info(layer->name, &layer->tops[0]);
}

void shape_rpn_bbox_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;

  // 3d tensor: C x H x W
  Tensor* const bbox = &layer->tops[0];

  // reshape to 4d tensor: (C / 4) x 4 x H x W
  bbox->ndim = 4;
  for (int n = 0; n < bbox->num_items; ++n) {
    const int C = bbox->shape[n][0];
    const int H = bbox->shape[n][1];
    const int W = bbox->shape[n][2];
    bbox->shape[n][0] = C / 4;
    bbox->shape[n][1] = 4;
    bbox->shape[n][2] = H;
    bbox->shape[n][3] = W;
  }
}

void forward_rcnn_pred_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;

  Tensor* const pred = &layer->tops[0];
  const Tensor* const score = layer->p_bottoms[0];

  pred->ndim = 4;
  pred->num_items = 1;
  pred->shape[0][0] = score->shape[0][0];
  pred->shape[0][1] = score->shape[0][1];
  pred->shape[0][2] = 1;
  pred->shape[0][3] = 1;
  pred->start[0] = 0;

  softmax_inplace_forward(pred, net->temp_data);

  shape_rcnn_pred_layer(net, layer);

  print_tensor_info(layer->name, &layer->tops[0]);
}

void shape_rcnn_pred_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;

  Tensor* const pred = &layer->tops[0];
  const Tensor* const score = layer->p_bottoms[0];
  const Tensor* const roi = layer->p_bottoms[1];

  pred->ndim = 2;
  pred->num_items = roi->num_items;
  {
    int total_size = 0;
    for (int n = 0; n < roi->num_items; ++n) {
      pred->shape[n][0] = roi->shape[n][0];
      pred->shape[n][1] = score->shape[0][1];
      pred->start[n] = total_size;
      total_size += pred->shape[n][0] * pred->shape[n][1];
    }
  }
}

void forward_rcnn_bbox_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;

  shape_rcnn_bbox_layer(net_, layer);

  print_tensor_info(layer->name, &layer->tops[0]);
}

void shape_rcnn_bbox_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;

  Tensor* const bbox = &layer->tops[0];
  const Tensor* const roi = layer->p_bottoms[1];

  bbox->ndim = 3;
  bbox->num_items = roi->num_items;
  {
    const int out_channels = bbox->shape[0][1];
    int total_size = 0;
    for (int n = 0; n < roi->num_items; ++n) {
      bbox->shape[n][0] = roi->shape[n][0];
      bbox->shape[n][1] = out_channels / 4;
      bbox->shape[n][2] = 4;
      bbox->start[n] = total_size;
      total_size += roi->shape[n][0] * out_channels;
    }
  }
}
