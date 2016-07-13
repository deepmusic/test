#include "layer.h"

// --------------------------------------------------------------------------
// kernel code
//   transform_box: transform a box according to a given gradient
//   enumerate_output: enumerate all RCNN box outputs
// --------------------------------------------------------------------------

// transform a box according to a given gradient
//   box: (x1, y1, x2, y2)
//   gradient: dx, dy, d(log w), d(log h)
#ifdef GPU
__device__
#endif
static
void transform_box(real box[],
                   const real dx, const real dy,
                   const real d_log_w, const real d_log_h,
                   const real img_W, const real img_H)
{
  // width & height of box
  const real w = box[2] - box[0] + 1.0f;
  const real h = box[3] - box[1] + 1.0f;
  // center location of box
  const real ctr_x = box[0] + 0.5f * w;
  const real ctr_y = box[1] + 0.5f * h;

  // new center location according to gradient (dx, dy)
  const real pred_ctr_x = dx * w + ctr_x;
  const real pred_ctr_y = dy * h + ctr_y;
  // new width & height according to gradient d(log w), d(log h)
  const real pred_w = exp(d_log_w) * w;
  const real pred_h = exp(d_log_h) * h;

  // update upper-left corner location
  box[0] = pred_ctr_x - 0.5f * pred_w;
  box[1] = pred_ctr_y - 0.5f * pred_h;
  // update lower-right corner location
  box[2] = pred_ctr_x + 0.5f * pred_w;
  box[3] = pred_ctr_y + 0.5f * pred_h;

  // adjust new corner locations to be within the image region,
  box[0] = MAX(0.0f,  MIN(box[0],  img_W - 1.0f));
  box[1] = MAX(0.0f,  MIN(box[1],  img_H - 1.0f));
  box[2] = MAX(0.0f,  MIN(box[2],  img_W - 1.0f));
  box[3] = MAX(0.0f,  MIN(box[3],  img_H - 1.0f));
}

// enumerate all output boxes for each object class
// and resize boxes to raw image size
#ifdef GPU
__global__
static
void enumerate_output_gpu(const real bottom2d[],
                          const real d_anchor3d[],
                          const real roi2d[],
                          const int num_rois, const int num_classes,
                          const real img_H, const real img_W,
                          const real scale_H, const real scale_W,
                          real top2d[])
{
  // index = c * num_rois + r
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num_rois * num_classes) {
    const int r = index / num_classes;
    const int c = index % num_classes;

    const real* const p_d_anchor3d = d_anchor3d + index * 4;
    const real dx = p_d_anchor3d[0];
    const real dy = p_d_anchor3d[1];
    const real d_log_w = p_d_anchor3d[2];
    const real d_log_h = p_d_anchor3d[3];

    const real* const p_roi2d = roi2d + r * 5;
    real* const p_top2d = top2d + index * 6;
    p_top2d[0] = c;
    p_top2d[1] = p_roi2d[0];
    p_top2d[2] = p_roi2d[1];
    p_top2d[3] = p_roi2d[2];
    p_top2d[4] = p_roi2d[3];
    p_top2d[5] = bottom2d[index];

    transform_box(p_top2d + 1, dx, dy, d_log_w, d_log_h, img_W, img_H);

    // resize box to raw image size
    p_top2d[1] /= scale_W;
    p_top2d[2] /= scale_H;
    p_top2d[3] /= scale_W;
    p_top2d[4] /= scale_H;
  }
}
#else
static
void enumerate_output_cpu(const real bottom2d[],
                          const real d_anchor3d[],
                          const real roi2d[],
                          const int num_rois, const int num_classes,
                          const real img_H, const real img_W,
                          const real scale_H, const real scale_W,
                          real top2d[])
{
  for (int r = 0; r < num_rois; ++r) {
    for (int c = 0; c < num_classes; ++c) {
      const int index = r * num_classes + c;
      const real* const p_d_anchor3d = d_anchor3d + index * 4;
      const real dx = p_d_anchor3d[0];
      const real dy = p_d_anchor3d[1];
      const real d_log_w = p_d_anchor3d[2];
      const real d_log_h = p_d_anchor3d[3];

      const real* const p_roi2d = roi2d + r * 5;
      real* const p_top2d = top2d + index * 6;
      p_top2d[0] = c;
      p_top2d[1] = p_roi2d[0];
      p_top2d[2] = p_roi2d[1];
      p_top2d[3] = p_roi2d[2];
      p_top2d[4] = p_roi2d[3];
      p_top2d[5] = bottom2d[index];

      transform_box(p_top2d + 1, dx, dy, d_log_w, d_log_h, img_W, img_H);

      p_top2d[1] /= scale_W;
      p_top2d[2] /= scale_H;
      p_top2d[3] /= scale_W;
      p_top2d[4] /= scale_H;
    }
  }
}
#endif



// --------------------------------------------------------------------------
// layer operator code
//   odtest_forward
// --------------------------------------------------------------------------

void odtest_forward(const Tensor* const bottom2d,
                    const Tensor* const d_anchor3d,
                    const Tensor* const roi2d,
                    const Tensor* const img_info1d,
                    Tensor* const top2d,
                    const LayerOption* const option)
{
  // do forward-pass for each item in the batch
  const real* p_bottom_item = bottom2d->data;
  const real* p_d_anchor_item = d_anchor3d->data;
  const real* p_roi_item = roi2d->data;
  const real* p_img_info = img_info1d->data;
  real* p_top_item = top2d->data;
  for (int n = 0; n < bottom2d->num_items; ++n) {
    const int num_rois = bottom2d->shape[n][0];
    const int num_classes = bottom2d->shape[n][1];

    // input image height & width
    const real img_H = p_img_info[0];
    const real img_W = p_img_info[1];
    // scale factor for height & width
    const real scale_H = p_img_info[2];
    const real scale_W = p_img_info[3];

    // enumerate all RCNN box outputs ("num_rois * num_classes" outputs)
    #ifdef GPU
    {
      const int num_threads = num_rois * num_classes;
      const int threads_per_block = 256;
      const int num_blocks = DIV_THEN_CEIL(num_threads,  threads_per_block);

      enumerate_output_gpu<<<num_blocks, threads_per_block>>>(
          p_bottom_item,  p_d_anchor_item,  p_roi_item,
          num_rois,  num_classes,  img_H,  img_W,  scale_H,  scale_W,
          p_top_item);
    }
    #else
    {
      enumerate_output_cpu(
          p_bottom_item,  p_d_anchor_item,  p_roi_item,
          num_rois,  num_classes,  img_H,  img_W,  scale_H,  scale_W,
          p_top_item);
    }
    #endif

    // set top shape: (num_rois * num_classes) x 6
    //   (class index, x1, y1, x2, y2, score) for each output
    top2d->shape[n][0] = num_rois * num_classes;
    top2d->shape[n][1] = 6;

    // locate next item
    {
      const int bottom_size = num_rois * num_classes;
      const int d_anchor_size = bottom_size * 4;
      const int roi_size = num_rois * 5;
      const int img_info_size = 6;
      const int top_size = bottom_size * 6;
      p_bottom_item += bottom_size;
      p_d_anchor_item += d_anchor_size;
      p_roi_item += roi_size;
      p_img_info += img_info_size;
      p_top_item += top_size;
    }
  } // endfor batch

  top2d->ndim = 2;
  top2d->num_items = bottom2d->num_items;
  {
    int total_size = 0;
    for (int n = 0; n < bottom2d->num_items; ++n) {
      const int top_size = top2d->shape[n][0] * top2d->shape[n][1];
      top2d->start[n] = total_size;
      total_size += top_size;
    }
  }
}



// --------------------------------------------------------------------------
// layer shape calculator code
// --------------------------------------------------------------------------

void odtest_shape(const Tensor* const bottom2d,
                  Tensor* const top2d,
                  const LayerOption* const option)
{
  int total_num_rois = 0;

  // calculate shape for each item in the batch
  top2d->ndim = 2;
  top2d->num_items = bottom2d->num_items;
  for (int n = 0; n < bottom2d->num_items; ++n) {
    const int num_rois = bottom2d->shape[n][0];
    const int num_classes = bottom2d->shape[n][1];

    // calculate total number of RoIs for determining temporary space size
    total_num_rois += num_rois * num_classes;

    // top shape = (num_rois * num_classes) x 6
    //   (class index, x1, y1, x2, y2, score) for each output
    top2d->shape[n][0] = num_rois * num_classes;
    top2d->shape[n][1] = 6;
    top2d->start[n] = total_num_rois * 6;
  }
}



// --------------------------------------------------------------------------
// API code
// --------------------------------------------------------------------------

void forward_odtest_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;

  odtest_forward(layer->p_bottoms[0], layer->p_bottoms[1],
                 layer->p_bottoms[2], layer->p_bottoms[3],
                 layer->p_tops[0],
                 &layer->option);

  print_tensor_info(layer->name, layer->p_tops[0]);
}

void shape_odtest_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;

  odtest_shape(layer->p_bottoms[0], layer->p_tops[0],
               &layer->option);

  update_net_size(net, layer, 0, 0, 0);
}

