#include "layer.h"

// --------------------------------------------------------------------------
// kernel code
//   scale_const_{gpu, cpu}
//   scale_channel_{gpu, cpu}
//   scale_channel_nobias_{gpu, cpu}
// --------------------------------------------------------------------------

// linear transform bottom -> top with constant weight and bias
//   top[i] = bottom[i] * weight + bias
#ifdef GPU
__global__
void scale_const_gpu(const real bottom[], real top[],
                     const real weight, const real bias,
                     const long int data_size)
{
  const long int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < data_size) {
    top[index] = bottom[index] * weight + bias;
  }
}
#else
void scale_const_cpu(const real bottom[], real top[],
                     const real weight, const real bias,
                     const long int data_size)
{
  for (long int index = 0; index < data_size; ++index) {
    top[index] = bottom[index] * weight + bias;
  }
}
#endif

// linear transform with channel-wise constants
//   top[c][i] = bottom[c][i] * weight[c] + bias[c]
#ifdef GPU
__global__
void scale_channel_gpu(const real bottom[], real top[],
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
void scale_channel_cpu(const real bottom[], real top[],
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

// linear transform with channel-wise constants, without bias
//   top[c][i] = bottom[c][i] * weight[c]
#ifdef GPU
__global__
void scale_channel_nobias_gpu(const real bottom[], real top[],
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
void scale_channel_nobias_cpu(const real bottom[], real top[],
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
// layer operator code
//   scale_const_forward
//   scale_channel_forward
// --------------------------------------------------------------------------

void scale_const_forward(const Tensor* const bottom,
                         Tensor* const top,
                         const LayerOption* const option)
{
  const long int data_size = flatten_size(bottom);
  const real weight = option->scale_weight;
  const real bias = (option->bias) ? option->scale_bias : 0;

  #ifdef GPU
  {
    const int threads_per_block = 512;
    const int num_blocks = DIV_THEN_CEIL(data_size,  threads_per_block);
    scale_const_gpu<<<num_blocks, threads_per_block>>>(
        bottom->data,  top->data,  weight,  bias,  data_size);
  }
  #else
  {
    scale_const_cpu(
        bottom->data,  top->data,  weight,  bias,  data_size);
  }
  #endif
}

void scale_channel_forward(const Tensor* const bottom,
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
        scale_channel_gpu<<<num_blocks, threads_per_block>>>(
            p_bottom_item,  p_top_item,  weight->data,  bias->data,  C,  D);
      }
      else {
        scale_channel_nobias_gpu<<<num_blocks, threads_per_block>>>(
            p_bottom_item,  p_top_item,  weight->data,  C,  D);
      }
    }
    #else
    {
      if (option->bias) {
        scale_channel_cpu(
            p_bottom_item,  p_top_item,  weight->data,  bias->data,  C,  D);
      }
      else {
        scale_channel_nobias_cpu(
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
// layer shape calculator code
// --------------------------------------------------------------------------

void scale_const_shape(const Tensor* const bottom,
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

void scale_channel_shape(const Tensor* const bottom,
                         Tensor* const top,
                         Tensor* const weight,
                         Tensor* const bias,
                         const LayerOption* const option)
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

  scale_const_shape(bottom, top);
} 



// --------------------------------------------------------------------------
// API code
// --------------------------------------------------------------------------

void forward_scale_const_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;

  scale_const_forward(layer->p_bottoms[0], layer->p_tops[0], &layer->option);
}

void forward_scale_channel_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;
  Tensor* const p_bias = (layer->option.bias) ? layer->p_params[1] : NULL;

  scale_channel_forward(layer->p_bottoms[0], layer->p_tops[0],
                        layer->p_params[0], p_bias,
                        &layer->option);
}

void shape_scale_const_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;

  scale_const_shape(layer->p_bottoms[0], layer->p_tops[0]);
}

void shape_scale_channel_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;
  Tensor* const p_bias = (layer->option.bias) ? layer->p_params[1] : NULL;

  scale_channel_shape(layer->p_bottoms[0], layer->p_tops[0],
                      layer->p_params[0], p_bias,
                      &layer->option);
}
