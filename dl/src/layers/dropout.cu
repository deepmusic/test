#include "layers/operator.h"
#include <string.h>
#include <limits.h>

// --------------------------------------------------------------------------
// kernel code
//   dropout_{gpu, cpu}
//   dropout_scaled_{gpu, cpu}
//   dropout_test_{gpu, cpu}
// --------------------------------------------------------------------------

// dropout transform bottom -> top
//   uint_thresh = UINT_MAX * dropout_ratio
//   top[i] = bottom[i] if mask[i] > uint_thresh, otherwise 0
#ifdef GPU
__global__
static
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
static
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
//   uint_thresh = UINT_MAX * dropout_ratio
//   inv_scale = 1 / (1 - dropout_ratio)
//   top[i] = inv_scale * bottom[i] if mask[i] > uint_thresh
#ifdef GPU
__global__
static
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
static
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
//   scale = 1 - dropout_ratio
//   top[i] = scale * bottom[i]
#ifdef GPU
__global__
static
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
static
void dropout_test_cpu(const real bottom[], real top[],
                      const long int data_size,
                      const real scale)
{
  for (long int index = 0; index < data_size; ++index) {
    top[index] = scale * bottom[index];
  }
}
#endif



// --------------------------------------------------------------------------
// layer-wise operator code
// --------------------------------------------------------------------------

// dropout transform: bottom -> top
//   if option->scaled_dropout = 1, perform scaled dropout
//   if option->test_dropout = 1, perform testing-time dropout
//   if both = 1, perform testing-time scaled dropout,
//                which is actually do nothing:  top[i] = bottom[i]
//   if both = 0, perform dropout
//   data size: total number of nodes (N * C * H * W or something)
//   mask: data_size x 1 temporary array
static
void dropout_forward(const Tensor* const bottom,
                     unsigned int mask[],
                     Tensor* const top,
                     const LayerOption* const option)
{
  const long int data_size = get_data_size(bottom);

  // perform dropout transform
  #ifdef GPU
  {
    const int threads_per_block = 512;
    const int num_blocks = DIV_THEN_CEIL(data_size,  threads_per_block);
    if (option->test_dropout) {
      if (!option->scaled_dropout) {
        // testing-time dropout
        dropout_test_gpu<<<num_blocks, threads_per_block>>>(
            bottom->data,  top->data,  data_size,
            1.0f - option->dropout_ratio);
      }
      else if (top->data != bottom->data) {
        // testing-time scaled dropout  (= no operation)
        cudaMemcpyAsync(top->data, bottom->data, data_size * sizeof(real),
                        cudaMemcpyDeviceToDevice);
      }
      #ifdef DEBUG
      else {
        printf("%s -> %s: No dropout operation\n", bottom->name, top->name);
      }
      #endif
    }
    else {
      // TODO: random number generation

      unsigned int uint_thresh
          = (unsigned int)option->dropout_ratio * UINT_MAX;
      if (!option->scaled_dropout) {
        // dropout
        dropout_gpu<<<num_blocks, threads_per_block>>>(
            bottom->data,  mask,  top->data,  data_size,  uint_thresh);
      }
      else {
        // scaled dropout
        dropout_scaled_gpu<<<num_blocks, threads_per_block>>>(
            bottom->data,  mask,  top->data,  data_size,  uint_thresh,
            1.0f / (1.0f - option->dropout_ratio));
      }
    }
  }
  #else
  {
    if (option->test_dropout) {
      if (!option->scaled_dropout) {
        // testing-time dropout
        dropout_test_cpu(
            bottom->data,  top->data,  data_size,
            1.0f - option->dropout_ratio);
      }
      else if (top->data != bottom->data) {
        // testing-time scaled dropout  (= no operation)
        memcpy(top->data, bottom->data, data_size * sizeof(real));
      }
      #ifdef DEBUG
      else {
        printf("%s -> %s: No dropout operation\n", bottom->name, top->name);
      }
      #endif
    }
    else {
      // TODO: random number generation

      unsigned int uint_thresh
          = (unsigned int)option->dropout_ratio * UINT_MAX;
      if (!option->scaled_dropout) {
        // dropout
        dropout_cpu(
            bottom->data,  mask,  top->data,  data_size,  uint_thresh);
      }
      else {
        // scaled dropout
        dropout_scaled_cpu(
            bottom->data,  mask,  top->data,  data_size,  uint_thresh,
            1.0f / (1.0f - option->dropout_ratio));
      }
    }
  }
  #endif
}



// --------------------------------------------------------------------------
// output shape calculator code
// --------------------------------------------------------------------------

static
void dropout_shape(const Tensor* const bottom,
                   Tensor* const top,
                   long int* const p_temp_space)
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

  *p_temp_space = get_data_size(bottom) * sizeof(real);
}



// --------------------------------------------------------------------------
// functions for layer instance
// --------------------------------------------------------------------------

void forward_dropout_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;
  dropout_forward(get_bottom(layer, 0), (unsigned int*)net->temp_data,
                  get_top(layer, 0), &layer->option);
}

void shape_dropout_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;
  long int temp_space;

  dropout_shape(get_bottom(layer, 0), get_top(layer, 0),
                &temp_space);

  update_temp_space(net, temp_space);
}

void init_dropout_layer(void* const net_, void* const layer_)
{
  return;
}

void free_dropout_layer(void* const net_, void* const layer_)
{
  return;
}
