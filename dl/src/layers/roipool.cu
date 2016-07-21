#include "core/net.h"

// --------------------------------------------------------------------------
// kernel code
//   roi_pool_{gpu, cpu}
// --------------------------------------------------------------------------

// RoI pooling bottom3d (C x H x W) -> top4d (R x C x H' x W')
//   given pixel (r, c, h, w) at top4d and RoI (x1, y1,, x2, y2),
//     top4d[r][c][h][w] = max_{hb,wb}{ bottom3d[c][hb][wb] }
//       hb, wb: pooling region corresponding to (h, w)
#ifdef GPU
__global__
static
void roi_pool_gpu(const real bottom3d[], const real roi2d[],
                  real top4d[],
                  const int R, const int C, const int H, const int W,
                  const int top_H, const int top_W,
                  const real spatial_scale)
{
  // thread index: (r, c, h, w) = r*C*H'*W' + c*H'*W' + h*W' + w
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < R * C * top_W * top_H) {
    // parse thread index -> (r, c, h, w)
    const int r = index / top_W / top_H / C;
    const int c = (index / top_W / top_H) % C;
    const int h = (index / top_W) % top_H;
    const int w = index % top_W;

    // RoI in the bottom plane
    const int x1 = ROUND(roi2d[r * 5 + 0] * spatial_scale);
    const int y1 = ROUND(roi2d[r * 5 + 1] * spatial_scale);
    const int x2 = ROUND(roi2d[r * 5 + 2] * spatial_scale);
    const int y2 = ROUND(roi2d[r * 5 + 3] * spatial_scale);
    const int roi_W = x2 - x1 + 1;
    const int roi_H = y2 - y1 + 1;

    // pooling region for pixel top[r][c][h][w]
    const int hb_start = MIN(H,  MAX(0,
                           y1 + (h * roi_H) / top_H));
    const int hb_end = MIN(H,  MAX(0,
                           y1 + DIV_THEN_CEIL((h + 1) * roi_H,  top_H)));
    const int wb_start = MIN(W,  MAX(0,
                           x1 + (w * roi_W) / top_W));
    const int wb_end = MIN(W,  MAX(0,
                           x1 + DIV_THEN_CEIL((w + 1) * roi_W,  top_W)));

    // find maximum in the bottom region
    const real* p_bottom3d = bottom3d + c * H * W;
    int maxidx = hb_start * W + wb_start;
    real maxval = p_bottom3d[maxidx];
    for (int hb = hb_start; hb < hb_end; ++hb) {
      for (int wb = wb_start; wb < wb_end; ++wb) {
        const int bottom_index = hb * W + wb;
        if (p_bottom3d[bottom_index] > maxval) {
          maxval = p_bottom3d[bottom_index];
          maxidx = bottom_index;
        }
      }
    }

    // if the bottom region is not empty,
    //   top[r][c][h][w] = "max in the region"
    // otherwise, assign 0
    {
      const int not_empty = (hb_start < hb_end) * (wb_start < wb_end);
      top4d[index] = not_empty * maxval;
    }
  }
}
#else
static
void roi_pool_cpu(const real bottom3d[], const real roi2d[],
                  real top4d[],
                  const int R, const int C, const int H, const int W,
                  const int top_H, const int top_W,
                  const real spatial_scale)
{
  const int top_area = top_H * top_W;
  const int top_volume = C * top_H * top_W;

  for (int r = 0; r < R; ++r) {
    // RoI in the bottom plane
    const int x1 = ROUND(roi2d[r * 5 + 0] * spatial_scale);
    const int y1 = ROUND(roi2d[r * 5 + 1] * spatial_scale);
    const int x2 = ROUND(roi2d[r * 5 + 2] * spatial_scale);
    const int y2 = ROUND(roi2d[r * 5 + 3] * spatial_scale);
    const int roi_W = x2 - x1 + 1;
    const int roi_H = y2 - y1 + 1;

    for (int h = 0; h < top_H; ++h) {
    for (int w = 0; w < top_W; ++w) {
      // pooling region for pixel top[r][c][h][w]
      const int hb_start = MIN(H,  MAX(0,
                               y1 + (h * roi_H) / top_H));
      const int hb_end = MIN(H,  MAX(0,
                             y1 + DIV_THEN_CEIL((h + 1) * roi_H,  top_H)));
      const int wb_start = MIN(W,  MAX(0,
                               x1 + (w * roi_W) / top_W));
      const int wb_end = MIN(W,  MAX(0,
                             x1 + DIV_THEN_CEIL((w + 1) * roi_W,  top_W)));

      const int top_index = r * top_volume + h * top_W + w;

      // if the bottom region is empty,
      //   top[r][c][h][w] = 0
      if (hb_start >= hb_end || wb_start >= wb_end) {
        for (int c = 0; c < C; ++c) {
          top4d[top_index + c * top_area] = 0;
        }
        continue;
      }

      // if the bottom region is not empty,
      //   top[r][c][h][w] = "max in the region"
      for (int c = 0; c < C; ++c) {
        // find maximum in the bottom region
        const real* p_bottom3d = bottom3d + c * H * W;
        int maxidx = hb_start * W + wb_start;
        for (int hb = hb_start; hb < hb_end; ++hb) {
          for (int wb = wb_start; wb < wb_end; ++wb) {
            maxidx = (p_bottom3d[hb * W + wb] > p_bottom3d[maxidx]) ?
                      hb * W + wb : maxidx;
          }
        }
        top4d[top_index + c * top_area] = p_bottom3d[maxidx];
      } // endfor c
    }} // endfor h, w
  } // endfor r
}
#endif



// --------------------------------------------------------------------------
// layer-wise operator code
// --------------------------------------------------------------------------

// RoI pooling: bottom -> top
//   bottom: C x H x W
//   roi: R x 5
//   top: R x C x H' x W'
static
void roipool_forward(const Tensor* const bottom3d,
                     const Tensor* const roi2d,
                     Tensor* const top4d,
                     const LayerOption* option)
{
  // top height & width
  const int top_H = option->pooled_height; // H'
  const int top_W = option->pooled_width; // W'

  // do forward-pass for each item in the batch
  const real* p_bottom_item = bottom3d->data;
  const real* p_roi_item = roi2d->data;
  real* p_top_item = top4d->data;
  for (int n = 0; n < bottom3d->num_items; ++n) {
    // bottom shape: R x C x H X W
    const int R = roi2d->shape[n][0];
    const int C = bottom3d->shape[n][0];
    const int H = bottom3d->shape[n][1];
    const int W = bottom3d->shape[n][2];

    // set top shape: R x C x H' x W'
    top4d->shape[n][0] = R;
    top4d->shape[n][1] = C;
    top4d->shape[n][2] = top_H;
    top4d->shape[n][3] = top_W;

    // RoI pooling
    //   bottom3d (C x H x W) -> top4d (R x C x H' x W')
    {
    #ifdef GPU
      const int num_threads = R * C * top_H * top_W;
      const int threads_per_block = 512;
      const int num_blocks = DIV_THEN_CEIL(num_threads,  threads_per_block);
      roi_pool_gpu<<<num_blocks, threads_per_block>>>(
          p_bottom_item,  p_roi_item,  p_top_item,
          R,  C,  H,  W,  top_H,  top_W,  option->spatial_scale);
    #else
      roi_pool_cpu(
          p_bottom_item,  p_roi_item,  p_top_item,
          R,  C,  H,  W,  top_H,  top_W,  option->spatial_scale);
    #endif
    }

    // locate next item
    {
      const int bottom_size = C * H * W;
      const int roi_size = R * 5;
      const int top_size = R * C * top_H * top_W;
      p_bottom_item += bottom_size;
      p_roi_item += roi_size;
      p_top_item += top_size;
    }
  } // endfor batch

  // if option->flatten_shape = true,
  // reshape to 2d tensor: total_num_rois x (C * H' * W')
  if (option->flatten_shape) {
    // for all items, C should be equal to each other
    const int C = bottom3d->shape[0][0];

    // calculate total number of RoI-pooled data
    int total_num_rois = 0;
    for (int n = 0; n < roi2d->num_items; ++n) {
      total_num_rois += roi2d->shape[n][0];
    }

    // reshape to 2d tensor: total_num_rois x (C * H' * W')
    top4d->ndim = 2;
    top4d->num_items = 1;
    top4d->shape[0][0] = total_num_rois;
    top4d->shape[0][1] = C * top_H * top_W;
    top4d->start[0] = 0;
  }
  else {
    top4d->ndim = 4;
    top4d->num_items = bottom3d->num_items;
    {
      int total_size = 0;
      for (int n = 0; n < bottom3d->num_items; ++n) {
        const int R = roi2d->shape[n][0];
        const int C = bottom3d->shape[n][0];
        const int top_size = R * C * top_H * top_W;
        top4d->start[n] = total_size;
        total_size += top_size;
      }
    }
  }
}



// --------------------------------------------------------------------------
// output shape calculator code
// --------------------------------------------------------------------------

static
void roipool_shape(const Tensor* const bottom3d,
                   const Tensor* const roi2d,
                   Tensor* const top4d,
                   const LayerOption* option)
{
  // top height & width
  const int top_H = option->pooled_height; // H'
  const int top_W = option->pooled_width; // W'

  // if option->flatten_shape = true,
  // reshape to 2d tensor: total_num_rois x (C * H' * W')
  if (option->flatten_shape) {
    // for all items, C should be equal to each other
    const int C = bottom3d->shape[0][0];

    // calculate total number of RoI-pooled data
    int total_num_rois = 0;
    for (int n = 0; n < roi2d->num_items; ++n) {
      total_num_rois += roi2d->shape[n][0];
    }

    // reshape to 2d tensor: total_num_rois x (C * H' * W')
    top4d->ndim = 2;
    top4d->num_items = 1;
    top4d->shape[0][0] = total_num_rois;
    top4d->shape[0][1] = C * top_H * top_W;
    top4d->start[0] = 0;

    return;
  }

  // otherwise, calculate shape for each item in the batch
  for (int n = 0; n < bottom3d->num_items; ++n) {
    // bottom shape: R x C x H X W
    const int R = roi2d->shape[n][0];
    const int C = bottom3d->shape[n][0];

    // top shape: R x C x H' x W'
    top4d->shape[n][0] = R;
    top4d->shape[n][1] = C;
    top4d->shape[n][2] = top_H;
    top4d->shape[n][3] = top_W;
  }

  top4d->ndim = 4;
  top4d->num_items = bottom3d->num_items;
  {
    int total_size = 0;
    for (int n = 0; n < bottom3d->num_items; ++n) {
      const int R = roi2d->shape[n][0];
      const int C = bottom3d->shape[n][0];
      const int top_size = R * C * top_H * top_W;
      top4d->start[n] = total_size;
      total_size += top_size;
    }
  }
}



// --------------------------------------------------------------------------
// functions for layer instance
// --------------------------------------------------------------------------

void forward_roipool_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;
  roipool_forward(get_bottom(layer, 0), get_bottom(layer, 1),
                  get_top(layer, 0), &layer->option);
}

void shape_roipool_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;
  roipool_shape(get_bottom(layer, 0), get_bottom(layer, 1),
                get_top(layer, 0), &layer->option);
}

void init_roipool_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;

  // it commonly reduces memory consumption
  get_top(layer, 0)->data_type = PRIVATE_DATA;
}

void free_roipool_layer(void* const net_, void* const layer_)
{
  return;
}
