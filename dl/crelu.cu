#include "layer.h"
#include <string.h>

// --------------------------------------------------------------------------
// kernel code
//   minus_{gpu, cpu}
// --------------------------------------------------------------------------

// in-place negative transform bottom -> bottom
//   bottom[i] = -bottom[i]
#ifdef GPU
__global__
static
void minus_inplace_gpu(real bottom[], const int item_size)
{
  const long int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < item_size) {
    bottom[index] = -bottom[index];
  }
}
#else
static
void minus_inplace_cpu(real bottom[], const int item_size)
{
  for (int index = 0; index < item_size; ++index) {
    bottom[index] = -bottom[index];
  }
}
#endif



// --------------------------------------------------------------------------
// layer operator code
// --------------------------------------------------------------------------

static
void crelu_forward(const Tensor* const bottom,
                   Tensor* const top,
                   const LayerOption* const option)
{
  for (int n = bottom->num_items - 1; n >= 0; --n) {
    int item_size = 1;
    for (int i = 0; i < bottom->ndim; ++i) {
      item_size *= bottom->shape[n][i];
    }

    #ifdef GPU
    {
      const int threads_per_block = 512;
      const int num_blocks = DIV_THEN_CEIL(item_size,  threads_per_block);
      cudaMemcpyAsync(top->data + top->start[n] + item_size,
                      bottom->data + bottom->start[n],
                      item_size * sizeof(real),
                      cudaMemcpyDeviceToDevice);
      cudaMemcpyAsync(top->data + top->start[n],
                      bottom->data + bottom->start[n],
                      item_size * sizeof(real),
                      cudaMemcpyDeviceToDevice);
      minus_inplace_gpu<<<num_blocks, threads_per_block>>>(
          top->data + top->start[n] + item_size,
          item_size);
    }
    #else
    {
      memcpy(top->data + top->start[n] + item_size,
             bottom->data + bottom->start[n],
             item_size * sizeof(real));
      memcpy(top->data + top->start[n],
             bottom->data + bottom->start[n],
             item_size * sizeof(real));
      minus_inplace_cpu(top->data + top->start[n] + item_size,
                        item_size);
    }
    #endif
  }
}



// --------------------------------------------------------------------------
// layer shape calculator code
// --------------------------------------------------------------------------

static
void crelu_shape(const Tensor* const bottom,
                 Tensor* const top)
{
  top->ndim = bottom->ndim;
  top->num_items = bottom->num_items;
  for (int n = 0; n < bottom->num_items; ++n) {
    top->shape[n][0] = bottom->shape[n][0] * 2; // 2x channels
    for (int i = 1; i < bottom->ndim; ++i) {
      top->shape[n][i] = bottom->shape[n][i];
    }
  }
  for (int n = 0; n < bottom->num_items; ++n) {
    top->start[n] = bottom->start[n] * 2;
  }
}



// --------------------------------------------------------------------------
// API code
// --------------------------------------------------------------------------

void forward_crelu_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;

  crelu_forward(get_bottom(layer, 0), get_top(layer, 0), &layer->option);
}

void shape_crelu_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;

  crelu_shape(get_bottom(layer, 0), get_top(layer, 0));
}
