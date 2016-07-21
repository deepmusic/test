#include "layers/operator.h"

// --------------------------------------------------------------------------
// kernel code
//   scale_weight_bias_{gpu, cpu}
//   scale_weight_{gpu, cpu}
// --------------------------------------------------------------------------

// element-wise transform with channel-wise weight and bias
//   top[c][i] = bottom[c][i] * weight[c] + bias[c]
#ifdef GPU
__global__
static
void scale_weight_bias_gpu(const real bottom[], real top[],
                           const real weight[], const real bias[],
                           const int C, const int D)
{
  // thread index (c, d) = c * D + d
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < C * D) {
    const int c = index / D;
    top[index] = bottom[index] * weight[c] + bias[c];
  }
}
#else
static
void scale_weight_bias_cpu(const real bottom[], real top[],
                           const real weight[], const real bias[],
                           const int C, const int D)
{
  for (int c = 0; c < C; ++c) {
    for (int d = 0; d < D; ++d) {
      top[c * D + d] = bottom[c * D + d] * weight[c] + bias[c];
    }
  }
}
#endif

// element-wise transform with channel-wise weight
//   top[c][i] = bottom[c][i] * weight[c]
#ifdef GPU
__global__
static
void scale_weight_gpu(const real bottom[], real top[],
                      const real weight[],
                      const int C, const int D)
{
  // thread index (c, d) = c * D + d
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < C * D) {
    const int c = index / D;
    top[index] = bottom[index] * weight[c];
  }
}
#else
static
void scale_weight_cpu(const real bottom[], real top[],
                      const real weight[],
                      const int C, const int D)
{
  for (int c = 0; c < C; ++c) {
    for (int d = 0; d < D; ++d) {
      top[c * D + d] = bottom[c * D + d] * weight[c];
    }
  }
}
#endif



// --------------------------------------------------------------------------
// layer-wise operator code
// --------------------------------------------------------------------------

static
void scale_forward(const Tensor* const bottom,
                   Tensor* const top,
                   const Tensor* const weight,
                   const Tensor* const bias,
                   const LayerOption* const option)
{
  const int C = weight->shape[0][0];

  const real* p_bottom_item = bottom->data;
  real* p_top_item = top->data;

  for (int n = 0; n < bottom->num_items; ++n) {
    int D = 1;
    for (int i = 1; i < bottom->ndim; ++i) {
      D *= bottom->shape[n][i];
    }

    #ifdef GPU
    {
      const int threads_per_block = 512;
      const int num_blocks = DIV_THEN_CEIL(C * D,  threads_per_block);
      if (option->bias) {
        scale_weight_bias_gpu<<<num_blocks, threads_per_block>>>(
            p_bottom_item,  p_top_item,  weight->data,  bias->data,  C,  D);
      }
      else {
        scale_weight_gpu<<<num_blocks, threads_per_block>>>(
            p_bottom_item,  p_top_item,  weight->data,  C,  D);
      }
    }
    #else
    {
      if (option->bias) {
        scale_weight_bias_cpu(
            p_bottom_item,  p_top_item,  weight->data,  bias->data,  C,  D);
      }
      else {
        scale_weight_cpu(
            p_bottom_item,  p_top_item,  weight->data,  C,  D);
      }
    }
    #endif

    // locate next item
    {
      p_bottom_item += C * D;
      p_top_item += C * D;
    }
  }
}



// --------------------------------------------------------------------------
// output & parameter shape calculator code
// --------------------------------------------------------------------------

static
void scale_shape(const Tensor* const bottom,
                 Tensor* const top,
                 Tensor* const weight,
                 Tensor* const bias,
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

  // parameter shape
  {
    const int C = bottom->shape[0][0];

    // weight shape: C x 1
    weight->num_items = 1;
    weight->ndim = 1;
    weight->shape[0][0] = C;
    weight->start[0] = 0;

    // bias shape: C x 1
    if (option->bias) {
      bias->num_items = 1;
      bias->ndim = 1;
      bias->shape[0][0] = C;
      bias->start[0] = 0;
    }
  }
} 



// --------------------------------------------------------------------------
// functions for layer instance
// --------------------------------------------------------------------------

void forward_scale_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;
  Tensor* const p_bias = (layer->option.bias) ? get_param(layer, 1) : NULL;

  scale_forward(get_bottom(layer, 0), get_top(layer, 0),
                get_param(layer, 0), p_bias, &layer->option);
}

void shape_scale_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;
  Tensor* const p_bias = (layer->option.bias) ? get_param(layer, 1) : NULL;

  scale_shape(get_bottom(layer, 0), get_top(layer, 0),
              get_param(layer, 0), p_bias, &layer->option);
}

void init_scale_layer(void* const net_, void* const layer_)
{
  return;
}

void free_scale_layer(void* const net_, void* const layer_)
{
  return;
}
