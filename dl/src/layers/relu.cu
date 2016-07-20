#include "core/net.h"

// --------------------------------------------------------------------------
// kernel code
//   relu_{gpu, cpu}
//   prelu_{gpu, cpu}
// --------------------------------------------------------------------------

// ReLU transform bottom -> top
//   top[i] = 0 if bottom[i] <= 0
#ifdef GPU
__global__
static
void relu_gpu(const real bottom[], real top[],
              const long int data_size)
{
  const long int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < data_size) {
    top[index] = (bottom[index] > 0) ? bottom[index] : 0;
  }
}
#else
static
void relu_cpu(const real bottom[], real top[],
              const long int data_size)
{
  for (long int index = 0; index < data_size; ++index) {
    top[index] = (bottom[index] > 0) ? bottom[index] : 0;
  }
}
#endif

// soft ReLU transform bottom -> top
//   top[i] = slope * bottom[i] if bottom[i] <= 0
#ifdef GPU
__global__
static
void prelu_gpu(const real bottom[], real top[],
               const long int data_size, const real negative_slope)
{
  const long int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < data_size) {
    top[index] = (bottom[index] > 0) ? bottom[index] :
                                       bottom[index] * negative_slope;
  }
}
#else
static
void prelu_cpu(const real bottom[], real top[],
               const long int data_size, const real negative_slope)
{
  for (long int index = 0; index < data_size; ++index) {
    top[index] = (bottom[index] > 0) ? bottom[index] :
                                       bottom[index] * negative_slope;
  }
}
#endif



// --------------------------------------------------------------------------
// layer operator code
// --------------------------------------------------------------------------

// (soft-)ReLU transform: bottom -> top
//   data size: total number of nodes (N * C * H * W or something)
//   if option->negative_slope = 0, perform ReLU
//                             > 0, perform soft ReLU
void relu_forward(const Tensor* const bottom,
                  Tensor* const top,
                  const LayerOption* const option)
{
  const long int data_size = get_data_size(bottom);

  // perform (soft-)ReLU transform
  //   if option->negative_slope = 0, perform ReLU
  //                             > 0, perform soft ReLU
  #ifdef GPU
  {
    const int threads_per_block = 512;
    const int num_blocks = DIV_THEN_CEIL(data_size,  threads_per_block);
    if (option->negative_slope == 0) {
      relu_gpu<<<num_blocks, threads_per_block>>>(
          bottom->data,  top->data,  data_size);
    }
    else {
      prelu_gpu<<<num_blocks, threads_per_block>>>(
          bottom->data,  top->data,  data_size,  option->negative_slope);
    }
  }
  #else
  {
    if (option->negative_slope == 0) {
      relu_cpu(
          bottom->data,  top->data,  data_size);
    }
    else {
      prelu_cpu(
          bottom->data,  top->data,  data_size,  option->negative_slope);
    }
  }
  #endif
}



// --------------------------------------------------------------------------
// layer shape calculator code
// --------------------------------------------------------------------------

void relu_shape(const Tensor* const bottom,
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
// API code
// --------------------------------------------------------------------------

void forward_relu_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;

  relu_forward(get_bottom(layer, 0), get_top(layer, 0), &layer->option);
}

void shape_relu_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;

  relu_shape(get_bottom(layer, 0), get_top(layer, 0));
}
