#include "layer.h"

// --------------------------------------------------------------------------
// kernel code
//   max_pool_{gpu, cpu}
// --------------------------------------------------------------------------

// max-pooling bottom3d (C x H x W) -> top3d (C x H' x W')
//   given (c, h', w'),
//     top3d[c][h'][w'] = max_{h, w} bottom3d[c][h][w]
//     argmax3d[c][h'][w'] = argmax_{h, w} bottom3d[c][h][w]
//   for
//     h = (-pad_h + stride_h * h') + { 0, 1, ..., kernel_h - 1 }
//     w = (-pad_w + stride_w * w') + { 0, 1, ..., kernel_w - 1 }
#ifdef GPU
__global__
void max_pool_gpu(const real bottom3d[],
                  real top3d[],
                  int argmax3d[],
                  const int C, const int bottom_H, const int bottom_W,
                  const int top_H, const int top_W,
                  const int kernel_h, const int kernel_w,
                  const int pad_h, const int pad_w,
                  const int stride_h, const int stride_w)
{
  // thread index: (c, h', w') = c*H'*W' + h'*W' + w'
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < C * top_H * top_W) {
    // parse thread index -> (c, h', w')
    const int c = index / top_H / top_W;
    const int ht = (index / top_W) % top_H;
    const int wt = index % top_W; 

    // pooling range in bottom
    //   h = (-pad_h + stride_h * h') + { 0, 1, ..., kernel_h - 1}
    //   w = (-pad_w + stride_w * w') + { 0, 1, ..., kernel_w - 1}
    const int h_start = MAX(0,  -pad_h + stride_h * ht);
    const int w_start = MAX(0,  -pad_w + stride_w * wt);
    const int h_end = MIN(-pad_h + stride_h * ht + kernel_h,  bottom_H);
    const int w_end = MIN(-pad_w + stride_w * wt + kernel_w,  bottom_W);

    if (h_start >= h_end || w_start >= w_end) {
      top3d[index] = 0;
      argmax3d[index] = -1;
    }

    // find maximum in the pooling region
    const real* const p_bottom3d = bottom3d + c * bottom_H * bottom_W;
    int maxidx = h_start * bottom_W + w_start;
    real maxval = p_bottom3d[maxidx];
    for (int h = h_start; h < h_end; ++h) {
      for (int w = w_start; w < w_end; ++w) {
        if (p_bottom3d[h * bottom_W + w] > maxval) {
          maxidx = h * bottom_W + w;
          maxval = p_bottom3d[maxidx];
        }
      }
    }

    // if pooling region is not empty,
    //   top3d[c][h'][w'] = "max in the region"
    // otherwise, assign 0
    {
      const int not_empty = (h_start < h_end) * (w_start < w_end);
      top3d[index] = not_empty * maxval;
      argmax3d[index] = not_empty * maxidx + (1 - not_empty) * (-1);
    }
  }
}
#else
void max_pool_cpu(const real bottom3d[],
                  real top3d[],
                  int argmax3d[],
                  const int C, const int bottom_H, const int bottom_W,
                  const int top_H, const int top_W,
                  const int kernel_h, const int kernel_w,
                  const int pad_h, const int pad_w,
                  const int stride_h, const int stride_w)
{
  // thread index: (c, h', w') = c*H'*W' + h'*W' + w'
  for (int index = 0; index < C * top_H * top_W; ++index) {
    // parse thread index -> (c, h', w')
    const int c = index / top_H / top_W;
    const int ht = (index / top_W) % top_H;
    const int wt = index % top_W; 

    // pooling range in bottom
    //   h = (-pad_h + stride_h * h') + { 0, 1, ..., kernel_h - 1}
    //   w = (-pad_w + stride_w * w') + { 0, 1, ..., kernel_w - 1}
    const int h_start = MAX(0,  -pad_h + stride_h * ht);
    const int w_start = MAX(0,  -pad_w + stride_w * wt);
    const int h_end = MIN(-pad_h + stride_h * ht + kernel_h,  bottom_H);
    const int w_end = MIN(-pad_w + stride_w * wt + kernel_w,  bottom_W);

    if (h_start >= h_end || w_start >= w_end) {
      top3d[index] = 0;
      argmax3d[index] = -1;
    }

    // find maximum in the pooling region
    const real* const p_bottom3d = bottom3d + c * bottom_H * bottom_W;
    int maxidx = h_start * bottom_W + w_start;
    real maxval = p_bottom3d[maxidx];
    for (int h = h_start; h < h_end; ++h) {
      for (int w = w_start; w < w_end; ++w) {
        if (p_bottom3d[h * bottom_W + w] > maxval) {
          maxidx = h * bottom_W + w;
          maxval = p_bottom3d[maxidx];
        }
      }
    }

    // if pooling region is not empty,
    //   top3d[c][h'][w'] = "max in the region"
    // otherwise, assign 0
    {
      const int not_empty = (h_start < h_end) * (w_start < w_end);
      top3d[index] = not_empty * maxval;
      argmax3d[index] = not_empty * maxidx + (1 - not_empty) * (-1);
    }
  }
}
#endif



// --------------------------------------------------------------------------
// layer operator code
//   pool_forward
// --------------------------------------------------------------------------

// max-pooling: bottom -> top
//   bottom: C x H x W
//   top: C x H' x W'
//   argmax: C x H' x W' array
void pool_forward(const Tensor* const bottom3d,
                  Tensor* const top3d,
                  int argmax_data[],
                  const LayerOption* const option)
{
  // kernel size, padding size & stride size
  const int kernel_h = option->kernel_h;
  const int kernel_w = option->kernel_w;
  const int pad_h = option->pad_h;
  const int pad_w = option->pad_w;
  const int stride_h = option->stride_h;
  const int stride_w = option->stride_w;

  // do forward-pass for each item in the batch
  const real* p_bottom_item = bottom3d->data;
  real* p_top_item = top3d->data;
  int* p_argmax_item = argmax_data;
  for (int n = 0; n < bottom3d->num_items; ++n) {
    // bottom shape: C x H x W
    const int C = bottom3d->shape[n][0];  // C
    const int bottom_H = bottom3d->shape[n][1];  // H
    const int bottom_W = bottom3d->shape[n][2];  // W

    // set top shape: C x H' x W'
    //   H' = 1 + (H + 2*pad_h - kernel_h) / stride_h
    //   W' = 1 + (W + 2*pad_w - kernel_w) / stride_w
    const int top_H
        = 1 + DIV_THEN_CEIL(bottom_H + 2 * pad_h - kernel_h,  stride_h);
    const int top_W
        = 1 + DIV_THEN_CEIL(bottom_W + 2 * pad_w - kernel_w,  stride_w);
    top3d->shape[n][0] = C;
    top3d->shape[n][1] = top_H;
    top3d->shape[n][2] = top_W;

    // max-pooling
    //   bottom3d (C x H x W) -> top3d (C x H' x W')
    #ifdef GPU
    {
      const int num_threads = C * top_H * top_W;
      const int threads_per_block = 512;
      const int num_blocks = DIV_THEN_CEIL(num_threads,  threads_per_block);
      max_pool_gpu<<<num_blocks, threads_per_block>>>(
          p_bottom_item,  p_top_item,  p_argmax_item,
          C,  bottom_H,  bottom_W,  top_H,  top_W,
          kernel_h,  kernel_w,  pad_h,  pad_w,  stride_h,  stride_w);
    }
    #else
    {
      max_pool_cpu(
          p_bottom_item,  p_top_item,  p_argmax_item,
          C,  bottom_H,  bottom_W,  top_H,  top_W,
          kernel_h,  kernel_w,  pad_h,  pad_w,  stride_h,  stride_w);
    }
    #endif

    // locate next item
    {
      const int bottom_size = C * bottom_H * bottom_W;
      const int top_size = C * top_H * top_W;
      p_bottom_item += bottom_size;
      p_top_item += top_size;
      p_argmax_item += top_size;
    }
  } // endfor batch

  top3d->ndim = 3;
  top3d->num_items = bottom3d->num_items;
}



// --------------------------------------------------------------------------
// layer shape calculator code
// --------------------------------------------------------------------------
void pool_shape(const Tensor* const bottom3d,
                Tensor* const top3d,
                int* const argmax_size,
                const LayerOption* const option)
{
  const int kernel_h = option->kernel_h;
  const int kernel_w = option->kernel_w;
  const int pad_h = option->pad_h;
  const int pad_w = option->pad_w;
  const int stride_h = option->stride_h;
  const int stride_w = option->stride_w;

  // calculate shape for each item in the batch
  int total_size = 0;
  for (int n = 0; n < bottom3d->num_items; ++n) {
    // bottom shape: C x H x W
    const int C = bottom3d->shape[n][0];  // C
    const int bottom_H = bottom3d->shape[n][1];  // H
    const int bottom_W = bottom3d->shape[n][2];  // W

    // top shape: C x H' x W'
    //   H' = 1 + (H + 2*pad_h - kernel_h) / stride_h
    //   W' = 1 + (W + 2*pad_w - kernel_w) / stride_w
    const int top_H
        = 1 + DIV_THEN_CEIL(bottom_H + 2 * pad_h - kernel_h,  stride_h);
    const int top_W
        = 1 + DIV_THEN_CEIL(bottom_W + 2 * pad_w - kernel_w,  stride_w);
    top3d->shape[n][0] = C;
    top3d->shape[n][1] = top_H;
    top3d->shape[n][2] = top_W;

    // start position for n-th item in top3d->data
    top3d->start[n] = total_size;
    total_size += C * top_H * top_W;
  }
  top3d->ndim = 3;
  top3d->num_items = bottom3d->num_items;

  // argmax data size = top size
  *argmax_size = total_size;
}



// --------------------------------------------------------------------------
// API code
// --------------------------------------------------------------------------

void forward_pool_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;

  pool_forward(layer->p_bottoms[0], layer->p_tops[0],
               net->tempint_data, &layer->option);
}

void shape_pool_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;

  int tempint_size;

  pool_shape(layer->p_bottoms[0], layer->p_tops[0],
             &tempint_size, &layer->option);

  update_net_size(net, layer, 0, tempint_size, 0);
}
