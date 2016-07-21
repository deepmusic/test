#include "core/net.h"

// --------------------------------------------------------------------------
// kernel code
//   power_order1_{gpu, cpu}
//   power_{gpu, cpu}
// --------------------------------------------------------------------------

// element-wise transform bottom -> top with constant weight and bias
//   top[i] = bottom[i] * weight + bias
#ifdef GPU
__global__
static
void power_order1_gpu(const real bottom[], real top[],
                      const real weight, const real bias,
                      const long int data_size)
{
  const long int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < data_size) {
    top[index] = bottom[index] * weight + bias;
  }
}
#else
static
void power_order1_cpu(const real bottom[], real top[],
                      const real weight, const real bias,
                      const long int data_size)
{
  for (long int index = 0; index < data_size; ++index) {
    top[index] = bottom[index] * weight + bias;
  }
}
#endif

// element-wise transform bottom -> top with constant weight and bias
//   top[i] = bottom[i] * weight + bias
#ifdef GPU
__global__
static
void power_gpu(const real bottom[], real top[],
               const real weight, const real bias, const real order,
               const long int data_size)
{
  const long int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < data_size) {
    top[index] = pow(bottom[index] * weight + bias, order);
  }
}
#else
static
void power_cpu(const real bottom[], real top[],
               const real weight, const real bias, const real order,
               const long int data_size)
{
  for (long int index = 0; index < data_size; ++index) {
    top[index] = pow(bottom[index] * weight + bias, order);
  }
}
#endif



// --------------------------------------------------------------------------
// layer-wise operator code
// --------------------------------------------------------------------------

static
void power_forward(const Tensor* const bottom,
                   Tensor* const top,
                   const LayerOption* const option)
{
  const long int data_size = get_data_size(bottom);
  const real weight = option->power_weight;
  const real bias = (option->bias) ? option->power_bias : 0;
  const real order = option->power_order;

  #ifdef GPU
  {
    const int threads_per_block = 512;
    const int num_blocks = DIV_THEN_CEIL(data_size,  threads_per_block);
    if (order == 1.0f) {
      power_order1_gpu<<<num_blocks, threads_per_block>>>(
          bottom->data,  top->data,  weight,  bias,  data_size);
    }
    else {
      power_gpu<<<num_blocks, threads_per_block>>>(
          bottom->data,  top->data,  weight,  bias,  order,  data_size);
    }
  }
  #else
  {
    if (order == 1.0f) {
      power_order1_cpu(
          bottom->data,  top->data,  weight,  bias,  data_size);
    }
    else {
      power_cpu(
          bottom->data,  top->data,  weight,  bias,  order,  data_size);
    }
  }
  #endif
}



// --------------------------------------------------------------------------
// output shape calculator code
// --------------------------------------------------------------------------

void power_shape(const Tensor* const bottom,
                 Tensor* const top)
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
}



// --------------------------------------------------------------------------
// functions for layer instance
// --------------------------------------------------------------------------

void forward_power_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;
  power_forward(get_bottom(layer, 0), get_top(layer, 0),
                &layer->option);
}

void shape_power_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;
  power_shape(get_bottom(layer, 0), get_top(layer, 0));
}

void init_power_layer(void* const net_, void* const layer_)
{
  return;
}

void free_power_layer(void* const net_, void* const layer_)
{
  return;
}
