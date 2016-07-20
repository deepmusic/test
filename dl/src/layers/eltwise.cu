#include "core/net.h"
#include <string.h>

// --------------------------------------------------------------------------
// kernel code
//   scale_const_{gpu, cpu}
// --------------------------------------------------------------------------

// element-wise addition
//   top[i] = top[i] + bottom[i]
#ifdef GPU
__global__
static
void eltwise_add_gpu(const real bottom[], real top[],
                     const long int data_size)
{
  const long int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < data_size) {
    top[index] += bottom[index];
  }
}
#else
static
void eltwise_add_cpu(const real bottom[], real top[],
                     const long int data_size)
{
  for (long int index = 0; index < data_size; ++index) {
    top[index] += bottom[index];
  }
}
#endif


// --------------------------------------------------------------------------
// layer operator code
//   eltwise_sum_forward
// --------------------------------------------------------------------------

// element-wise sum: top = bottoms[0] + ... + bottoms[num_bottoms-1]
static
void eltwise_sum_forward(const Tensor* const bottoms[],
                         Tensor* const top,
                         const int num_bottoms)
{
  if (num_bottoms > 0) {
    const int data_size = get_data_size(bottoms[0]);

    #ifdef GPU
    {
      const int threads_per_block = 512;
      const int num_blocks = DIV_THEN_CEIL(data_size,  threads_per_block);
      cudaMemcpyAsync(top->data, bottoms[0]->data, data_size * sizeof(real),
                      cudaMemcpyDeviceToDevice);
      for (int m = 1; m < num_bottoms; ++m) {
        eltwise_add_gpu<<<num_blocks, threads_per_block>>>(
            bottoms[m]->data, top->data, data_size);
      }
    }
    #else
    {
      memcpy(top->data, bottoms[0]->data, data_size * sizeof(real));
      for (int m = 1; m < num_bottoms; ++m) {
        eltwise_add_cpu(bottoms[m]->data, top->data, data_size);
      }
    }
    #endif
  }
}



// --------------------------------------------------------------------------
// layer shape calculator code
// --------------------------------------------------------------------------

static
void eltwise_shape(const Tensor* const bottoms[],
                   Tensor* const top,
                   const int num_bottoms)
{
  const Tensor* const p_bottom = (num_bottoms > 0) ? bottoms[0] : NULL;

  // top shape = bottom shape
  if (p_bottom) {
    top->ndim = p_bottom->ndim;
    top->num_items = p_bottom->num_items;
    for (int n = 0; n < p_bottom->num_items; ++n) {
      for (int i = 0; i < p_bottom->ndim; ++i) {
        top->shape[n][i] = p_bottom->shape[n][i];
      }
    }
    for (int n = 0; n < p_bottom->num_items; ++n) {
      top->start[n] = p_bottom->start[n];
    }
  }
}



// --------------------------------------------------------------------------
// API code
// --------------------------------------------------------------------------

void forward_eltwise_sum_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;

  eltwise_sum_forward(layer->p_bottoms, get_top(layer, 0),
                      layer->num_bottoms);
}

void shape_eltwise_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;

  eltwise_shape(layer->p_bottoms, get_top(layer, 0),
                layer->num_bottoms);
}
