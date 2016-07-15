#include "layer.h"
#include <limits.h>

// --------------------------------------------------------------------------
// kernel code
//   dropout_{gpu, cpu}
//   dropout_scaled_{gpu, cpu}
//   dropout_test_{gpu, cpu}
//   dropout_inplace_{gpu, cpu}
//   dropout_scaled_inplace_{gpu, cpu}
//   dropout_test_inplace_{gpu, cpu}
// --------------------------------------------------------------------------

// dropout transform bottom -> top
//   uint_thresh = UINT_MAX * threshold
//   top[i] = bottom[i] if mask[i] > uint_thresh, otherwise 0
#ifdef GPU
__global__
void dropout_gpu(const real bottom[],
                 const unsigned int mask[],
                 real top[],
                 const long int data_size,
                 const unsigned int uint_thresh)
{
  const long int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < data_size) {
    top[index] = (mask[index] > uint_thresh) * bottom[index];
  }
}
#else
void dropout_cpu(const real bottom[],
                 const unsigned int mask[],
                 real top[],
                 const long int data_size,
                 const unsigned int uint_thresh)
{
  for (long int index = 0; index < data_size; ++index) {
    top[index] = (mask[index] > uint_thresh) * bottom[index];
  }
}
#endif

// scaled dropout transform
//   uint_thresh = UINT_MAX * threshold
//   inv_scale = 1 / (1 - threshold)
//   top[i] = inv_scale * bottom[i] if mask[i] > uint_thresh
#ifdef GPU
__global__
void dropout_scaled_gpu(const real bottom[],
                        const unsigned int mask[],
                        real top[],
                        const long int data_size,
                        const unsigned int uint_thresh,
                        const real inv_scale)
{
  const long int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < data_size) {
    top[index] = (mask[index] > uint_thresh) * inv_scale
                 * bottom[index];
  }
}
#else
void dropout_scaled_cpu(const real bottom[],
                        const unsigned int mask[],
                        real top[],
                        const long int data_size,
                        const unsigned int uint_thresh,
                        const real inv_scale)
{
  for (long int index = 0; index < data_size; ++index) {
    top[index] = (mask[index] > uint_thresh) * inv_scale
                 * bottom[index];
  }
}
#endif

// testing-time dropout transform
//   scale = 1 - threshold
//   top[i] = scale * bottom[i]
#ifdef GPU
__global__
void dropout_test_gpu(const real bottom[], real top[],
                      const long int data_size,
                      const real scale)
{
  const long int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < data_size) {
    top[index] = scale * bottom[index];
  }
}
#else
void dropout_test_cpu(const real bottom[], real top[],
                      const long int data_size,
                      const real scale)
{
  for (long int index = 0; index < data_size; ++index) {
    top[index] = scale * bottom[index];
  }
}
#endif

// in-place dropout transform
#ifdef GPU
__global__
void dropout_inplace_gpu(real bottom[],
                         const unsigned int mask[],
                         const long int data_size,
                         const unsigned int uint_thresh)
{
  const long int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < data_size) {
    bottom[index] *= (mask[index] > uint_thresh);
  }
}
#else
void dropout_inplace_cpu(real bottom[],
                         const unsigned int mask[],
                         const long int data_size,
                         const unsigned int uint_thresh)
{
  for (long int index = 0; index < data_size; ++index) {
    bottom[index] *= (mask[index] > uint_thresh);
  }
}
#endif

// in-place scaled dropout transform
#ifdef GPU
__global__
void dropout_scaled_inplace_gpu(real bottom[],
                                const unsigned int mask[],
                                const long int data_size,
                                const unsigned int uint_thresh,
                                const real inv_scale)
{
  const long int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < data_size) {
    bottom[index] *= (mask[index] > uint_thresh) * inv_scale;
  }
}
#else
void dropout_scaled_inplace_cpu(real bottom[],
                                const unsigned int mask[],
                                const long int data_size,
                                const unsigned int uint_thresh,
                                const real inv_scale)
{
  for (long int index = 0; index < data_size; ++index) {
    bottom[index] *= (mask[index] > uint_thresh) * inv_scale;
  }
}
#endif

// testing-time in-place dropout transform
#ifdef GPU
__global__
void dropout_test_inplace_gpu(real bottom[],
                              const long int data_size,
                              const real scale)
{
  const long int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < data_size) {
    bottom[index] *= scale;
  }
}
#else
void dropout_test_inplace_cpu(real bottom[],
                              const long int data_size,
                              const real scale)
{
  for (long int index = 0; index < data_size; ++index) {
    bottom[index] *= scale;
  }
}
#endif



// --------------------------------------------------------------------------
// layer operator code
//   dropout_forward
//   dropout_forward_inplace
// --------------------------------------------------------------------------

// dropout transform: bottom -> top
//   if option->scaled = 1, perform scaled dropout
//   if option->test = 1, perform testing-time dropout
//   if both = 1, perform testing-time scaled dropout,
//                which is actually do nothing:  top[i] = bottom[i]
//   if both = 0, perform dropout
//   data size: total number of nodes (N * C * H * W or something)
//   mask: data_size x 1 temporary array
void dropout_forward(const Tensor* const bottom,
                     unsigned int mask[],
                     Tensor* const top,
                     const LayerOption* const option)
{
  const long int data_size = flatten_size(bottom);

  // perform dropout transform
  #ifdef GPU
  {
    const int threads_per_block = 512;
    const int num_blocks = DIV_THEN_CEIL(data_size,  threads_per_block);
    if (option->test) {
      if (option->scaled) {
        // testing-time scaled dropout  (= no operation)
        cudaMemcpyAsync(top->data, bottom->data, data_size * sizeof(real),
                        cudaMemcpyDeviceToDevice);
      }
      else {
        // testing-time dropout
        dropout_test_gpu<<<num_blocks, threads_per_block>>>(
            bottom->data,  top->data,  data_size,  1.0f - option->threshold);
      }
    }
    else {
      // TODO: random number generation

      unsigned int uint_thresh = (unsigned int)option->threshold * UINT_MAX;
      if (option->scaled) {
        // scaled dropout
        dropout_scaled_gpu<<<num_blocks, threads_per_block>>>(
            bottom->data,  mask,  top->data,  data_size,  uint_thresh,
            1.0f / (1.0f - option->threshold));
      }
      else {
        // dropout
        dropout_gpu<<<num_blocks, threads_per_block>>>(
            bottom->data,  mask,  top->data,  data_size,  uint_thresh);
      }
    }
  }
  #else
  {
    if (option->test) {
      if (option->scaled) {
        // testing-time scaled dropout  (= no operation)
        for (int i = 0; i < data_size; ++i) {
          top->data[i] = bottom->data[i];
        }
      }
      else {
        // testing-time dropout
        dropout_test_cpu(
            bottom->data,  top->data,  data_size,  1.0f - option->threshold);
      }
    }
    else {
      // TODO: random number generation

      unsigned int uint_thresh = (unsigned int)option->threshold * UINT_MAX;
      if (option->scaled) {
        // scaled dropout
        dropout_scaled_cpu(
            bottom->data,  mask,  top->data,  data_size,  uint_thresh,
            1.0f / (1.0f - option->threshold));
      }
      else {
        // dropout
        dropout_cpu(
            bottom->data,  mask,  top->data,  data_size,  uint_thresh);
      }
    }
  }
  #endif

  // set top shape (= bottom shape)
  {
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
}

// in-place dropout transform: bottom -> bottom
void dropout_forward_inplace(Tensor* const bottom,
                             unsigned int mask[],
                             const LayerOption* const option)
{
  const long int data_size = flatten_size(bottom);

  // perform dropout transform
  #ifdef GPU
  {
    const int threads_per_block = 512;
    const int num_blocks = DIV_THEN_CEIL(data_size,  threads_per_block);
    if (option->test) {
      if (option->scaled) {
        // testing-time scaled dropout  (= no operation)
        return;
      }
      else {
        // testing-time dropout
        dropout_test_inplace_gpu<<<num_blocks, threads_per_block>>>(
            bottom->data,  data_size,  option->threshold);
      }
    }
    else {
      // TODO: random number generation

      unsigned int uint_thresh = (unsigned int)option->threshold * UINT_MAX;
      if (option->scaled) {
        // scaled dropout
        dropout_scaled_inplace_gpu<<<num_blocks, threads_per_block>>>(
            bottom->data,  mask,  data_size,  uint_thresh,
            1.0f / (1.0f - option->threshold));
      }
      else {
        // dropout
        dropout_inplace_gpu<<<num_blocks, threads_per_block>>>(
            bottom->data,  mask,  data_size,  uint_thresh);
      }
    }
  }
  #else
  {
    if (option->test) {
      if (option->scaled) {
        // testing-time scaled dropout  (= no operation)
        return;
      }
      else {
        // testing-time dropout
        dropout_test_inplace_cpu(
            bottom->data,  data_size,  option->threshold);
      }
    }
    else {
      // TODO: random number generation

      unsigned int uint_thresh = (unsigned int)option->threshold * UINT_MAX;
      if (option->scaled) {
        // scaled dropout
        dropout_scaled_inplace_cpu(
            bottom->data,  mask,  data_size,  uint_thresh,
            1.0f / (1.0f - option->threshold));
      }
      else {
        // dropout
        dropout_inplace_cpu(
            bottom->data,  mask,  data_size,  uint_thresh);
      }
    }
  }
  #endif
}



// --------------------------------------------------------------------------
// layer shape calculator code
// --------------------------------------------------------------------------

void dropout_shape(const Tensor* const bottom,
                   Tensor* const top)
{
  // top shape = bottom shape
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



// --------------------------------------------------------------------------
// API code
// --------------------------------------------------------------------------

void forward_dropout_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;

  dropout_forward(layer->p_bottoms[0], (unsigned int*)net->tempint_data,
                  layer->p_tops[0], &layer->option);
}

void forward_inplace_dropout_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;

  dropout_forward_inplace(layer->p_tops[0], (unsigned int*)net->tempint_data,
                          &layer->option);
}

void shape_dropout_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;

  dropout_shape(layer->p_bottoms[0], layer->p_tops[0]);
}
