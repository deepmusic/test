#include "layer.h"

#include <time.h>

static float a_time[8] = { 0, };
static clock_t tick0, tick1;

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
void scale_const_gpu(const real* const bottom,
                     real* const top,
                     const real weight,
                     const real bias,
                     const long int data_size)
{
  const long int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < data_size) {
    top[index] = bottom[index] * weight + bias;
  }
}
#else
void scale_const_cpu(const real* const bottom,
                     real* const top,
                     const real weight,
                     const real bias,
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
void scale_channel_gpu(const real* const bottom,
                       real* const top,
                       const real* const weight,
                       const real* const bias,
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
void scale_channel_cpu(const real* const bottom,
                       real* const top,
                       const real* const weight,
                       const real* const bias,
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
void scale_channel_nobias_gpu(const real* const bottom,
                              real* const top,
                              const real* const weight,
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
void scale_channel_nobias_cpu(const real* const bottom,
                              real* const top,
                              const real* const weight,
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

  tick0 = clock();

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

  tick1 = clock();
  a_time[5] = (float)(tick1 - tick0) / CLOCKS_PER_SEC;
  a_time[6] = 0;
  a_time[7] += (float)(tick1 - tick0) / CLOCKS_PER_SEC;
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

  tick0 = clock();

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

  tick1 = clock();
  a_time[5] = (float)(tick1 - tick0) / CLOCKS_PER_SEC;
  a_time[6] = 0;
  a_time[7] += (float)(tick1 - tick0) / CLOCKS_PER_SEC;
}



// --------------------------------------------------------------------------
// layer shape calculator code
// --------------------------------------------------------------------------

void scale_shape(const Tensor* const bottom,
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

void forward_scale_const_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;

  scale_const_forward(layer->p_bottoms[0], &layer->tops[0], &layer->option);
  print_tensor_info(layer->name, &layer->tops[0]);
  #ifdef DEBUG
  {
    for (int i = 0; i < 8; ++i) {
      printf("%4.2f\t", a_time[i] * 1000);
    }
    printf("\n");
  }
  #endif
}

void forward_scale_channel_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;
  Tensor* p_bias = (layer->option.bias) ? &layer->params[1] : NULL;

  scale_channel_forward(layer->p_bottoms[0], &layer->tops[0],
                        &layer->params[0], p_bias,
                        &layer->option);
  print_tensor_info(layer->name, &layer->tops[0]);
  #ifdef DEBUG
  {
    for (int i = 0; i < 8; ++i) {
      printf("%4.2f\t", a_time[i] * 1000);
    }
    printf("\n");
  }
  #endif
}

void forward_inplace_scale_const_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;

  scale_const_forward(&layer->tops[0], &layer->tops[0], &layer->option);
  print_tensor_info(layer->name, &layer->tops[0]);
  #ifdef DEBUG
  {
    for (int i = 0; i < 8; ++i) {
      printf("%4.2f\t", a_time[i] * 1000);
    }
    printf("\n");
  }
  #endif
}

void forward_inplace_scale_channel_layer(void* const net_,
                                         void* const layer_)
{
  Layer* const layer = (Layer*)layer_;
  Tensor* p_bias = (layer->option.bias) ? &layer->params[1] : NULL;

  scale_channel_forward(&layer->tops[0], &layer->tops[0],
                        &layer->params[0], p_bias,
                        &layer->option);
  print_tensor_info(layer->name, &layer->tops[0]);
  #ifdef DEBUG
  {
    for (int i = 0; i < 8; ++i) {
      printf("%4.2f\t", a_time[i] * 1000);
    }
    printf("\n");
  }
  #endif
}

void shape_scale_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;

  scale_shape(layer->p_bottoms[0], &layer->tops[0]);
}
