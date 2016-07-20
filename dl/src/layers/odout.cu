#include "core/net.h"
#include "layers/rpn.h"

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
int transform_box(real box[],
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

// discard boxes whose scores < threshold
//   list: num_boxes x 5 array,  (x1, y1, x2, y2, score) for each box
static
int filter_box(real list[], const int num_boxes, const real threshold)
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

// enumerate all output boxes for each object class
#ifdef GPU
__global__
static
void enumerate_output_gpu(const real bottom2d[],
                          const real d_anchor3d[],
                          const real roi2d[],
                          const int num_rois, const int num_classes,
                          const real img_H, const real img_W,
                          const real min_box_H, const real min_box_W,
                          const real scale_H, const real scale_W,
                          real proposals[])
{
  // index = c * num_rois + r
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num_rois * num_classes) {
    const int c = index / num_rois;
    const int r = index % num_rois;

    const real* const p_d_anchor3d = d_anchor3d + (r * num_classes + c) * 4;
    const real dx = p_d_anchor3d[0];
    const real dy = p_d_anchor3d[1];
    const real d_log_w = p_d_anchor3d[2];
    const real d_log_h = p_d_anchor3d[3];

    const real* const p_roi2d = roi2d + r * 5;
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
#else
static
void enumerate_output_cpu(const real bottom2d[],
                          const real d_anchor3d[],
                          const real roi2d[],
                          const int num_rois, const int num_classes,
                          const real img_H, const real img_W,
                          const real min_box_H, const real min_box_W,
                          const real scale_H, const real scale_W,
                          real proposals[])
{
  // skip background class (c = 0)
  for (int c = 1; c < num_classes; ++c) {
    for (int r = 0; r < num_rois; ++r) {
      const real* const p_d_anchor3d
          = d_anchor3d + (r * num_classes + c) * 4;
      const real dx = p_d_anchor3d[0];
      const real dy = p_d_anchor3d[1];
      const real d_log_w = p_d_anchor3d[2];
      const real d_log_h = p_d_anchor3d[3];

      const real* const p_roi2d = roi2d + r * 5;
      real* const p_proposals = proposals + (c * num_rois + r) * 5;
      p_proposals[0] = p_roi2d[0];
      p_proposals[1] = p_roi2d[1];
      p_proposals[2] = p_roi2d[2];
      p_proposals[3] = p_roi2d[3];

      p_proposals[4]
          = transform_box(p_proposals,
                          dx, dy, d_log_w, d_log_h,
                          img_W, img_H, min_box_W, min_box_H)
            * bottom2d[r * num_classes + c];
    } // endfor n
  } // endfor c
}
#endif

// retrieve boxes that are determined to be kept by NMS
#ifdef GPU
__global__
static
void retrieve_output_gpu(const real proposals[],
                         const int index_out[],
                         real top2d[],
                         const int num_output, const int num_rois,
                         const real scale_H, const real scale_W)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num_output) {
    const real* const p_proposals = proposals + index_out[index] * 5;
    real* const p_top2d = top2d + index * 6;
    const int predicted_class = index_out[index] / num_rois;

    p_top2d[0] = predicted_class;
    p_top2d[1] = p_proposals[0] / scale_W;
    p_top2d[2] = p_proposals[1] / scale_H;
    p_top2d[3] = p_proposals[2] / scale_W;
    p_top2d[4] = p_proposals[3] / scale_H;
    p_top2d[5] = p_proposals[4];
  }
}
#else
static
void retrieve_output_cpu(const real proposals[],
                         const int index_out[],
                         real top2d[],
                         const int num_output, const int num_rois,
                         const real scale_H, const real scale_W)
{
  for (int index = 0; index < num_output; ++index) {
    const real* const p_proposals = proposals + index_out[index] * 5;
    real* const p_top2d = top2d + index * 6;
    const int predicted_class = index_out[index] / num_rois;

    p_top2d[0] = predicted_class;
    p_top2d[1] = p_proposals[0] / scale_W;
    p_top2d[2] = p_proposals[1] / scale_H;
    p_top2d[3] = p_proposals[2] / scale_W;
    p_top2d[4] = p_proposals[3] / scale_H;
    p_top2d[5] = p_proposals[4];
  }
}
#endif

// retrieve boxes that can be considered as "unseen" object classes
//   1. they are not predicted as any of known object classes
//   2. their RPN objectness scores are high (>= score_thresh)
//   3. they are not much overlapped with output boxes (iou <= nms_thresh)
// CPU mode only
#ifdef GPU
#else
static
void retrieve_unknown_cpu(const real proposals[],
                          int index_out[],
                          const real roi2d[],
                          real top2d[],
                          const int num_output, const int num_rois,
                          int* const num_output_with_unknown,
                          const real scale_H, const real scale_W,
                          const real score_thresh, const real nms_thresh)
{
  int num_unknown = 0;

  for (int index = 0; index < num_rois; ++index) {
    const real* const p_roi2d = roi2d + index * 5;
    int is_selected = 1;

    if (p_roi2d[4] < score_thresh) {
      break;
    }

    for (int i = 0; i < num_output; ++i) {
      const real* const p_proposals = proposals + index_out[i] * 5;
      if (iou(p_proposals, p_roi2d) > nms_thresh) {
        is_selected = 0;
        break;
      }
    }
    for (int i = 0; i < num_unknown; ++i) {
      const real* const p_unknown = roi2d + index_out[num_output + i] * 5;
      if (iou(p_unknown, p_roi2d) > nms_thresh) {
        is_selected = 0;
        break;
      }
    }

    if (is_selected) {
      real* const p_top2d = top2d + (num_output + num_unknown) * 6;
      p_top2d[0] = 0;
      p_top2d[1] = p_roi2d[0] / scale_W;
      p_top2d[2] = p_roi2d[1] / scale_H;
      p_top2d[3] = p_roi2d[2] / scale_W;
      p_top2d[4] = p_roi2d[3] / scale_H;
      p_top2d[5] = p_roi2d[4] - 0.2f;
      index_out[num_output + num_unknown] = index;
      ++num_unknown;
    }
  }

  *num_output_with_unknown = num_output + num_unknown;
}
#endif

// remove duplicated boxes, and select final output boxes
static
void filter_output(const real bottom2d[],
                   const real d_anchor3d[],
                   const real roi2d[],
                   real top2d[],
                   int* const num_output,
                   void* const nms_aux_data,
                   real proposals_cpu[],
                   int index_out_cpu[],
                   real proposals_gpu[],
                   int index_out_gpu[],
                   const real img_H, const real img_W,
                   const real scale_H, const real scale_W,
                   const int num_rois, const int num_classes,
                   const int min_size,
                   const real score_thresh, const real nms_thresh,
                   const int bbox_vote, const real vote_thresh)
{
  // minimum box height & width (w.r.t. input image size)
  const real min_box_H = min_size * scale_H;
  const real min_box_W = min_size * scale_W;

  // enumerate all RCNN box outputs ("num_rois * num_classes" outputs)
  #ifdef GPU
  {
    const int num_threads = num_rois * num_classes;
    const int threads_per_block = 256;
    const int num_blocks = DIV_THEN_CEIL(num_threads,  threads_per_block);

    enumerate_output_gpu<<<num_blocks, threads_per_block>>>(
        bottom2d,  d_anchor3d,  roi2d,  num_rois,  num_classes,
        img_H,  img_W,  min_box_H,  min_box_W,  scale_H,  scale_W,
        proposals_gpu);

    cudaMemcpyAsync(proposals_cpu, proposals_gpu,
                    num_threads * 5 * sizeof(real),
                    cudaMemcpyDeviceToHost);
  }
  #else
  {
    enumerate_output_cpu(
        bottom2d,  d_anchor3d,  roi2d,  num_rois,  num_classes,
        img_H,  img_W,  min_box_H,  min_box_W,  scale_H,  scale_W,
        proposals_cpu);
  }
  #endif

  // for each class, choose outputs according to scores & overlap
  {
    int num_out = 0;
    // skip background class (c = 0)
    for (int c = 1; c < num_classes; ++c) {
      const int base = c * num_rois;
      real* const p_proposals_cpu = proposals_cpu + base * 5;
      int* const p_index_out_cpu = index_out_cpu + num_out;

      // filter-out boxes whose scores less than threshold
      const int num_pre_nms
          = filter_box(p_proposals_cpu, num_rois, score_thresh);

      // sort & NMS
      if (num_pre_nms > 1) {
        int num_post_nms = 0;
        sort_box(p_proposals_cpu, 0, num_pre_nms - 1, num_pre_nms);

        #ifdef GPU
        {
          cudaMemcpyAsync(proposals_gpu, p_proposals_cpu,
                          num_pre_nms * 5 * sizeof(real),
                          cudaMemcpyHostToDevice);

          nms(num_pre_nms,  proposals_gpu,  nms_aux_data,
              &num_post_nms,  p_index_out_cpu,  base,
              nms_thresh,  num_pre_nms,  bbox_vote,  vote_thresh);
        }
        #else
        {
          nms(num_pre_nms,  p_proposals_cpu,  nms_aux_data,
              &num_post_nms,  p_index_out_cpu,  base,
              nms_thresh,  num_pre_nms,  bbox_vote,  vote_thresh);
        }
        #endif

        num_out += num_post_nms;
      }
      else {
        p_index_out_cpu[0] = base;
        num_out += num_pre_nms;
      }
    }
    *num_output = num_out;
  }

  // retrieve final outputs & resize box to raw image size
  #ifdef GPU
  {
    const int threads_per_block = 16;
    const int num_blocks = DIV_THEN_CEIL(*num_output,  threads_per_block);

    cudaMemcpyAsync(proposals_gpu, proposals_cpu,
                    num_rois * num_classes * 5 * sizeof(real),
                    cudaMemcpyHostToDevice);
    cudaMemcpyAsync(index_out_gpu, index_out_cpu,
                    (*num_output) * sizeof(int),
                    cudaMemcpyHostToDevice);

    retrieve_output_gpu<<<num_blocks, threads_per_block>>>(
        proposals_gpu,  index_out_gpu,  top2d,  *num_output,  num_rois,
        scale_H,  scale_W);
  }
  #else
  {
    retrieve_output_cpu(
        proposals_cpu,  index_out_cpu,  top2d,  *num_output,  num_rois,
        scale_H,  scale_W);
    retrieve_unknown_cpu(
        proposals_cpu,  index_out_cpu,  roi2d,  top2d,
        *num_output,  num_rois,  num_output,
        scale_H,  scale_W,  0.9f,  0.1f);
  }
  #endif
}



// --------------------------------------------------------------------------
// layer operator code
// --------------------------------------------------------------------------

static
void odout_forward(const Tensor* const bottom2d,
                   const Tensor* const d_anchor3d,
                   const Tensor* const roi2d,
                   const Tensor* const img_info1d,
                   Tensor* const top2d,
                   void* const nms_aux_data,
                   unsigned char temp_cpu_data[],
                   unsigned char temp_gpu_data[],
                   const LayerOption* const option)
{
  // do forward-pass for each item in the batch
  const real* p_bottom_item = bottom2d->data;
  const real* p_d_anchor_item = d_anchor3d->data;
  const real* p_roi_item = roi2d->data;
  real* p_top_item = top2d->data;

  int* const index_out_cpu = (int*)&temp_cpu_data[0];
  real* const proposals_cpu =
      (real*)&temp_cpu_data[option->pre_nms_topn * sizeof(real)];
  int* const index_out_gpu = (int*)&temp_gpu_data[0];
  real* const proposals_gpu =
      (real*)&temp_gpu_data[option->pre_nms_topn * sizeof(real)];

  #ifdef GPU
  real img_info_cpu[BATCH_SIZE * 6];
  const real* p_img_info_cpu = img_info_cpu;
  cudaMemcpyAsync(img_info_cpu, img_info1d->data,
                  get_data_size(img_info1d) * sizeof(real),
                  cudaMemcpyDeviceToHost);
  #else
  const real* p_img_info_cpu = img_info1d->data;
  #endif

  // bottom2d and d_anchor3d are flatten such that
  // all batch items are channel-wise concatenated into one big item
  // thus, num_items and num_channels are taken from roi2d->shape,
  // and num_classes is taken from bottom2d->shape[0]
  // (bottom2d->shape[1, ...] are all zero)
  const int num_classes = bottom2d->shape[0][1];
  for (int n = 0; n < roi2d->num_items; ++n) {
    const int num_rois = roi2d->shape[n][0];
    // input image height & width
    const real img_H = p_img_info_cpu[0];
    const real img_W = p_img_info_cpu[1];
    // scale factor for height & width
    const real scale_H = p_img_info_cpu[2];
    const real scale_W = p_img_info_cpu[3];
    // number of final output boxes (will be computed below)
    int num_output = 0;

    // remove duplicated boxes, and select final output boxes
    {
      filter_output(
          p_bottom_item,  p_d_anchor_item,  p_roi_item,
          p_top_item,  &num_output,  nms_aux_data,
          proposals_cpu,  index_out_cpu,  proposals_gpu,  index_out_gpu,
          img_H,  img_W,  scale_H,  scale_W,  num_rois,  num_classes,
          option->min_size,  option->score_thresh,  option->nms_thresh,
          option->bbox_vote,  option->vote_thresh);

      // set top shape: num_output x 6
      //   (class index, x1, y1, x2, y2, score) for each output
      top2d->shape[n][0] = num_output;
      top2d->shape[n][1] = 6;
    }

    // locate next item
    {
      const int bottom_size = num_rois * num_classes;
      const int d_anchor_size = bottom_size * 4;
      const int roi_size = num_rois * 5;
      const int img_info_size = 6;
      const int top_size = num_output * 6;
      p_bottom_item += bottom_size;
      p_d_anchor_item += d_anchor_size;
      p_roi_item += roi_size;
      p_img_info_cpu += img_info_size;
      p_top_item += top_size;
    }
  } // endfor batch

  top2d->ndim = 2;
  top2d->num_items = roi2d->num_items;
  {
    int total_size = 0;
    for (int n = 0; n < roi2d->num_items; ++n) {
      const int top_size = top2d->shape[n][0] * top2d->shape[n][1];
      top2d->start[n] = total_size;
      total_size += top_size;
    }
  }
}



// --------------------------------------------------------------------------
// layer shape calculator code
// --------------------------------------------------------------------------

static
void odout_shape(const Tensor* const bottom2d,
                 const Tensor* const roi2d,
                 Tensor* const top2d,
                 long int* const p_temp_space,
                 const LayerOption* const option)
{
  int total_num_rois = 0;

  // calculate shape for each item in the batch
  // bottom2d and d_anchor3d are flatten such that
  // all batch items are channel-wise concatenated into one big item
  // thus, num_items and num_channels are taken from roi2d->shape,
  // and num_classes is taken from bottom2d->shape[0]
  // (bottom2d->shape[1, ...] are all zero)
  const int num_classes = bottom2d->shape[0][1];
  for (int n = 0; n < roi2d->num_items; ++n) {
    const int num_rois = roi2d->shape[n][0];

    // calculate total number of RoIs for determining temporary space size
    total_num_rois += num_rois * num_classes;

    // top shape <= (num_rois * num_classes) x 6
    //   (class index, x1, y1, x2, y2, score) for each output
    //   exact number of outputs will be determined after forward-pass
    top2d->shape[n][0] = num_rois * num_classes;
    top2d->shape[n][1] = 6;
    top2d->start[n] = total_num_rois * 6;
  }
  top2d->ndim = 2;
  top2d->num_items = roi2d->num_items;

  // temporary space size
  {
    const int proposals_size = total_num_rois;
    const int index_out_size = total_num_rois;

    *p_temp_space = proposals_size * sizeof(real)
                    + index_out_size * sizeof(int);
  }
}



// --------------------------------------------------------------------------
// API code
// --------------------------------------------------------------------------

void forward_odout_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;

  odout_forward(get_bottom(layer, 0), get_bottom(layer, 1),
                get_bottom(layer, 2), get_bottom(layer, 3),
                get_top(layer, 0),
                layer->aux_data,
                (unsigned char*)net->temp_cpu_data,
                (unsigned char*)net->temp_data,
                &layer->option);
}

void shape_odout_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;
  long int temp_space;

  odout_shape(get_bottom(layer, 0), get_bottom(layer, 2),
              get_top(layer, 0),
              &temp_space, &layer->option);

  update_temp_space(net, temp_space);
}

void malloc_odout_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;
  long int space_cpu, space_gpu;

  malloc_nms_aux_data(&layer->aux_data, layer->option.pre_nms_topn,
                      &space_cpu, &space_gpu);

  net->space += space_gpu;
  net->space_cpu += space_cpu;

  #ifdef DEBUG
  {
    #ifdef GPU
    printf("%s: Memory allocated, CPU %ld byte and GPU %ld byte\n",
           layer->name, space_cpu, space_gpu);
    #else
    printf("%s: Memory allocated, %ld byte\n",
           layer->name, space_cpu + space_gpu);
    #endif
  }
  #endif
}

void free_odout_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;

  free_nms_aux_data(layer->aux_data);
}
