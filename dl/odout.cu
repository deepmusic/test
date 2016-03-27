#include "layer.h"
#include <math.h>
#include <stdio.h>

// --------------------------------------------------------------------------
// kernel code
//   transform_box: transform a box according to a given gradient
//   sort_box: sort a list of boxes in descending order of their scores
//   filter_box: discard boxes whose scores < threshold
//   filter_output: remove duplicated boxes, and select final output boxes
// --------------------------------------------------------------------------

// transform a box according to a given gradient
//   box: (x1, y1, x2, y2)
//   gradient: dx, dy, d(log w), d(log h)
#ifdef GPU
__device__
#endif
static
int transform_box(real* const box,
                  const real dx, const real dy,
                  const real d_log_w, const real d_log_h,
                  const real img_W, const real img_H,
                  const real min_box_W, const real min_box_H)
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

  // recompute new width & height
  const real box_w = box[2] - box[0] + 1.0f;
  const real box_h = box[3] - box[1] + 1.0f;

  // check if new box's size >= threshold
  return (box_w >= min_box_W) * (box_h >= min_box_H);
}

// quick-sort a list of boxes in descending order of their scores,
//   list: num_boxes x 5 array,  (x1, y1, x2, y2, score) for each box
static
void sort_box(real* const list, const int start, const int end)
{
  const real pivot_score = list[start * 5 + 4];
  int left = start + 1, right = end;
  real temp[5];
  while (left <= right) {
    while (left <= end && list[left * 5 + 4] >= pivot_score) ++left;
    while (right > start && list[right * 5 + 4] <= pivot_score) --right;
    if (left <= right) {
      for (int i = 0; i < 5; ++i) {
        temp[i] = list[left * 5 + i];
      }
      for (int i = 0; i < 5; ++i) {
        list[left * 5 + i] = list[right * 5 + i];
      }
      for (int i = 0; i < 5; ++i) {
        list[right * 5 + i] = temp[i];
      }
      ++left;
      --right;
    }
  }

  if (right > start) {
    for (int i = 0; i < 5; ++i) {
      temp[i] = list[start * 5 + i];
    }
    for (int i = 0; i < 5; ++i) {
      list[start * 5 + i] = list[right * 5 + i];
    }
    for (int i = 0; i < 5; ++i) {
      list[right * 5 + i] = temp[i];
    }
  }

  if (start < right - 1) {
    sort_box(list, start, right - 1);
  }
  if (right + 1 < end) {
    sort_box(list, right + 1, end);
  }
}

// discard boxes whose scores < threshold
//   list: num_boxes x 5 array,  (x1, y1, x2, y2, score) for each box
static
int filter_box(real* const list, const int num_boxes, const real threshold)
{
  int left = 0, right = num_boxes - 1;
  while (left <= right) {
    while (left < num_boxes && list[left * 5 + 4] >= threshold) ++left;
    while (right >= 0 && list[right * 5 + 4] < threshold) --right;
    if (left <= right) {
      for (int i = 0; i < 5; ++i) {
        list[left * 5 + i] = list[right * 5 + i];
      }
      ++left;
      --right;
    }
  }

  return left;
}

#ifdef GPU
__global__
void enumerate_output_gpu(const real* const bottom2d,
                          const real* const d_anchor3d,
                          const real* const roi2d,
                          const int num_rois, const int num_classes,
                          const real img_H, const real img_W,
                          const real min_box_H, const real min_box_W,
                          real* const proposals)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = index / num_rois;
  const int r = index % num_rois;
  if (index < num_rois * num_classes) {
    const real* const p_d_anchor3d = d_anchor3d + (r * num_classes + c) * 4;
    const real dx = p_d_anchor3d[0];
    const real dy = p_d_anchor3d[1];
    const real d_log_w = p_d_anchor3d[2];
    const real d_log_h = p_d_anchor3d[3];

    const real* const p_roi2d = roi2d + r * 4;
    real* const p_proposals = proposals + index * 5;
    p_proposals[0] = p_roi2d[0];
    p_proposals[1] = p_roi2d[1];
    p_proposals[2] = p_roi2d[2];
    p_proposals[3] = p_roi2d[3];

    p_proposals[4]
        = transform_box(p_proposals,
                        dx, dy, d_log_w, d_log_h,
                        img_W, img_H, min_box_W, min_box_H)
          * bottom2d[r * num_classes + c];
  }
}
#endif

#ifdef GPU
__global__
void retrieve_output_gpu(const real* const proposals,
                         const int* const keep,
                         real* const top2d,
                         const int num_output, const int num_rois,
                         const real x_scale, const real y_scale)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num_output) {
    const real* const p_proposals = proposals + keep[index] * 5;
    real* const p_top2d = top2d + index * 6;
    const int c = keep[index] / num_rois;

    p_top2d[0] = c;
    p_top2d[1] = p_proposals[0] / x_scale;
    p_top2d[2] = p_proposals[1] / y_scale;
    p_top2d[3] = p_proposals[2] / x_scale;
    p_top2d[4] = p_proposals[3] / y_scale;
    p_top2d[5] = p_proposals[4];
  }
}
#endif

// remove duplicated boxes, and select final output boxes
#ifdef GPU
void filter_output_gpu(const real* const bottom2d,
                       const real* const d_anchor3d,
                       const real* const roi2d,
                       const real* const img_info1d,
                       real* const top2d,
                       int* num_output,
                       real* const proposals,
                       int* const keep,
                       real* const proposals_dev,
                       int* const keep_dev,
                       const int num_rois, const int num_classes,
                       const int min_size,
                       const real score_thresh, const real nms_thresh)
{
  // enumerate all RCNN box outputs ("num_rois * num_classes" outputs)
  {
    // input image height & width
    const real img_H = img_info1d[0];
    const real img_W = img_info1d[1];
    // minimum box width & height
    const real min_box_W = min_size * img_info1d[2];
    const real min_box_H = min_size * img_info1d[3];

    const int num_threads = num_rois * num_classes;
    const int threads_per_block = 256;
    const int num_blocks = DIV_THEN_CEIL(num_threads,  threads_per_block);
    enumerate_output_gpu<<<num_blocks, threads_per_block>>>(
        bottom2d,  d_anchor3d,  roi2d,  num_rois,  num_classes,
        img_H,  img_W,  min_box_H,  min_box_W,
        proposals_dev);

    cudaMemcpyAsync(proposals, proposals_dev,
                    num_threads * 5 * sizeof(real),
                    cudaMemcpyDeviceToHost);
  }

  // for each class, choose outputs according to scores & overlap
  {
    int num_out = 0;
    for (int c = 1; c < num_classes; ++c) {
      const int base = c * num_rois;
      real* const p_proposals = proposals + base * 5;
      int* const p_keep = keep + num_out;

      // filter-out boxes whose scores less than threshold
      const int num_pre_nms
          = filter_box(p_proposals, num_rois, score_thresh);

      // sort & NMS
      if (num_pre_nms > 1) {
        int num_post_nms = 0;
        sort_box(p_proposals, 0, num_pre_nms - 1);
        nms(num_pre_nms,  p_proposals,  &num_post_nms,  p_keep,  base,
            nms_thresh,  num_pre_nms);
        num_out += num_post_nms;
      }
      else {
        p_keep[0] = base;
        num_out += num_pre_nms;
      }
    }
    *num_output = num_out;

    cudaMemcpyAsync(proposals_dev, proposals,
                    num_rois * num_classes * 5 * sizeof(real),
                    cudaMemcpyHostToDevice);
    cudaMemcpyAsync(keep_dev, keep,
                    (*num_output) * sizeof(int),
                    cudaMemcpyHostToDevice);
  }

  // retrieve final outputs & resize box to raw image size
  {
    const int threads_per_block = 16;
    const int num_blocks = DIV_THEN_CEIL(*num_output,  threads_per_block);
    retrieve_output_gpu<<<num_blocks, threads_per_block>>>(
        proposals_dev,  keep_dev,  top2d,  *num_output,  num_rois,
        img_info1d[2], img_info1d[3]);
  }
}
#else
void filter_output_cpu(const real* const bottom2d,
                       const real* const d_anchor3d,
                       const real* const roi2d,
                       const real* const img_info1d,
                       real* const top2d,
                       int* num_output,
                       real* const proposals,
                       int* const keep,
                       const int num_rois, const int num_classes,
                       const int min_size,
                       const real score_thresh, const real nms_thresh)
{
  // input image height & width
  const real img_H = img_info1d[0];
  const real img_W = img_info1d[1];
  // minimum box width & height
  const real min_box_W = min_size * img_info1d[2];
  const real min_box_H = min_size * img_info1d[3];

  // do for each object class (skip background class)
  real* p_top2d = top2d;
  for (int c = 1; c < num_classes; ++c) {
    // enumerate all RCNN box outputs ("num_rois" outputs for each class)
    for (int r = 0; r < num_rois; ++r) {
      const int index = r * num_classes + c;
      const real dx = d_anchor3d[index * 4 + 0];
      const real dy = d_anchor3d[index * 4 + 1];
      const real d_log_w = d_anchor3d[index * 4 + 2];
      const real d_log_h = d_anchor3d[index * 4 + 3];

      proposals[r * 5 + 0] = roi2d[r * 4 + 0];
      proposals[r * 5 + 1] = roi2d[r * 4 + 1];
      proposals[r * 5 + 2] = roi2d[r * 4 + 2];
      proposals[r * 5 + 3] = roi2d[r * 4 + 3];

      proposals[r * 5 + 4]
          = transform_box(&proposals[r * 5],
                          dx, dy, d_log_w, d_log_h,
                          img_W, img_H, min_box_W, min_box_H)
            * bottom2d[index];
    }

    // choose outputs according to scores & overlap
    {
      // filter-out boxes whose scores less than threshold
      const int num_pre_nms = filter_box(proposals, num_rois, score_thresh);
      int num_post_nms = 0;

      // sort & NMS
      if (num_pre_nms > 1) {
        sort_box(proposals, 0, num_pre_nms - 1);
        nms(num_pre_nms,  proposals,  &num_post_nms,  keep,  0,
            nms_thresh,  num_pre_nms);
      }
      else {
        num_post_nms = num_pre_nms;
        keep[0] = 0;
      }

      // retrieve final outputs & resize box to raw image size
      for (int r = 0; r < num_post_nms; ++r) {
        p_top2d[r * 6 + 0] = c;
        p_top2d[r * 6 + 1] = proposals[keep[r] * 5 + 0] / img_info1d[2];
        p_top2d[r * 6 + 2] = proposals[keep[r] * 5 + 1] / img_info1d[3];
        p_top2d[r * 6 + 3] = proposals[keep[r] * 5 + 2] / img_info1d[2];
        p_top2d[r * 6 + 4] = proposals[keep[r] * 5 + 3] / img_info1d[3];
        p_top2d[r * 6 + 5] = proposals[keep[r] * 5 + 4];
      }

      *num_output += num_post_nms;
      p_top2d += num_post_nms * 6;
    }
  } // endfor class
}
#endif



// --------------------------------------------------------------------------
// layer operator code
//   odout_forward
// --------------------------------------------------------------------------

void odout_forward(const Tensor* const bottom2d,
                   const Tensor* const d_anchor3d,
                   const Tensor* const roi2d,
                   const Tensor* const img_info1d,
                   Tensor* const top2d,
                   real* const proposals,
                   int* const keep,
                   real* const proposals_dev,
                   int* const keep_dev,
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
    int num_output = 0;

    {
    #ifdef GPU
      filter_output_gpu(
          p_bottom_item,  p_d_anchor_item,  p_roi_item,  p_img_info,
          p_top_item,  &num_output,
          proposals,  keep,  proposals_dev,  keep_dev,
          num_rois,  num_classes,
          option->min_size,  option->score_thresh,  option->nms_thresh);
    #else
      filter_output_cpu(
          p_bottom_item,  p_d_anchor_item,  p_roi_item,  p_img_info,
          p_top_item,  &num_output,
          proposals,  keep,
          num_rois,  num_classes,
          option->min_size,  option->score_thresh,  option->nms_thresh);
    #endif

      // set top shape: num_output x 6
      //   (class index, x1, y1, x2, y2, score) for each output
      top2d->shape[n][0] = num_output;
      top2d->shape[n][1] = 6;
    }

    // locate next item
    {
      const int bottom_size = num_rois * num_classes;
      const int d_anchor_size = bottom_size * 4;
      const int roi_size = num_rois * 4;
      const int img_info_size = 4;
      const int top_size = num_output * 6;
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

void odout_shape(const Tensor* const bottom2d,
                 Tensor* const top2d,
                 int* const proposals_size,
                 int* const keep_size,
                 const LayerOption* const option)
{
  int total_num_rois = 0;

  // calculate shape for each item in the batch
  top2d->ndim = 2;
  top2d->num_items = bottom2d->num_items;
  for (int n = 0; n < bottom2d->num_items; ++n) {
    // calculate total number of RoIs for determining temporary space size
    const int num_rois = bottom2d->shape[n][0];
    total_num_rois += num_rois;

    // top shape <= num_rois x 6
    //   (class index, x1, y1, x2, y2, score) for each output
    //   exact number of outputs will be determined after forward-pass
    top2d->shape[n][0] = num_rois;
    top2d->shape[n][1] = 6;
    top2d->start[n] = num_rois * 6;
  }

  // temporary space size
  //   in GPU mode, total space allocated for proposals should be
  //   a power of 2 >= actual number of proposals
  {
    int num_power_of_2 = 1;
    while (num_power_of_2 < total_num_rois) num_power_of_2 *= 2;
    *proposals_size = num_power_of_2 * 5;
    *keep_size = total_num_rois;
  }
}



// --------------------------------------------------------------------------
// API code
// --------------------------------------------------------------------------

void forward_odout_layer(Net* const net, Layer* const layer)
{
  odout_forward(layer->p_bottoms[0], layer->p_bottoms[1],
                layer->p_bottoms[2], layer->p_bottoms[3],
                &layer->tops[0],
                net->temp_cpu_data, net->tempint_cpu_data,
                net->temp_data, net->tempint_data,
                &layer->option);

  print_tensor_info(layer->name, &layer->tops[0]);
}

void shape_odout_layer(Net* const net, Layer* const layer)
{
  int temp_size, tempint_size;

  odout_shape(layer->p_bottoms[0], &layer->tops[0],
              &temp_size, &tempint_size, &layer->option);

  update_net_size(net, layer, temp_size, tempint_size, 0);
}
