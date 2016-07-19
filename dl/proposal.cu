/*
  Original version (25.8ms)
    1. [1ms] memcpy, D->H
      1-1. scores (75*2*36*46*float = 993.6KB)
      1-2. bbox (75*4*36*46*float = 1987.2KB)
    2. [15ms] all candidate enumeration & sort
    3. [0ms] memcpy, H->D, 6000*5*float = 120KB
    4. [3.3ms] nms kernel
    5. [1.8ms] memcpy, D->H, 6000*94*uint64 = 4512KB
    6. [0.7ms] nms post processing (bitwise calculations)
    7. [4ms] roi -> top

  Improved version (6.3ms)
    1. [0ms] no memcpy required
    2. [2.6ms] all candidate enumeration & sort
      2-1. [0.3ms] all candidate enumeration
      2-2. [0.6ms] memcpy, D->H, all candidates (75*36*46*5*float = 2484KB)
      2-3. [1.6ms] partial quick-sort
    3. [0ms] memcpy, H->D, 6000*5*float = 120KB
    4. [1.1ms] nms kernel
    5. [1.8ms] memcpy, D->H, 6000*94*uint64 = 4512KB
    6. [0.7ms] nms post processing
    7. [0.1ms] roi -> top

  TODO
    - GPU sort (improve 2-2, 2-3) - speedup
    - GPU nms post processing (remove 5)
*/

#include "layer.h"
#include <string.h>

#include <time.h>

static float a_time[8] = { 0, };
static clock_t tick0, tick1, tick00;

// --------------------------------------------------------------------------
// kernel code
//   generate_anchors: generate anchor boxes of varying sizes and ratios
//   transform_box: transform a box according to a given gradient
//   sort_box: sort a list of boxes in descending order of their scores
//   enumerate_proposals: generate all candidate boxes with their scores
//   retrieve_rois: retrieve boxes that are determined to be kept by NMS
// --------------------------------------------------------------------------

// given a base box, enumerate transformed boxes of varying sizes and ratios
//   base_size: base box's width & height (i.e., base box is square)
//   scales: "num_scales x 1" array of scale factors for base box transform
//   ratios: "num_ratios x 1" array of height-width ratios
//   anchors: "num_anchors x 4" array,  (x1, y1, x2, y2) for each box
//   num_anchors: total number of transformations
//                = num_scales * num_ratios
static
void generate_anchors(const real scales[], const real ratios[],
                      real anchors[],
                      const int num_scales, const int num_ratios,
                      const int base_size)
{
  // base box's width & height & center location
  const real base_area = (real)(base_size * base_size);
  const real center = 0.5f * (base_size - 1.0f);

  // enumerate all transformed boxes
  {
    real* p_anchors = anchors;
    for (int j0 = 0; j0 < num_scales; j0 += num_ratios) {
    for (int i = 0; i < num_ratios; ++i) {
      // transformed width & height for given ratio factors
      const real ratio_w = (real)ROUND(sqrt(base_area / ratios[i]));
      const real ratio_h = (real)ROUND(ratio_w * ratios[i]);

      for (int j = 0; j < num_ratios; ++j) {
        // transformed width & height for given scale factors
        const real scale_w = 0.5f * (ratio_w * scales[j0 + j] - 1.0f);
        const real scale_h = 0.5f * (ratio_h * scales[j0 + j] - 1.0f);

        // (x1, y1, x2, y2) for transformed box
        p_anchors[0] = center - scale_w;
        p_anchors[1] = center - scale_h;
        p_anchors[2] = center + scale_w;
        p_anchors[3] = center + scale_h;

        p_anchors += 4;
      } // endfor j
    }} // endfor i, j0
  }
}

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

// quick-sort a list of boxes in descending order of their scores (CPU)
//   list_cpu: num_boxes x 5 array,  (x1, y1, x2, y2, score) for each box
//             located at main memory
//   if num_top <= end,  only top-k results are guaranteed to be sorted
//   (for efficient computation)
void sort_box(real list_cpu[], const int start, const int end,
              const int num_top)
{
  const real pivot_score = list_cpu[start * 5 + 4];
  int left = start + 1, right = end;
  real temp[5];
  while (left <= right) {
    while (left <= end && list_cpu[left * 5 + 4] >= pivot_score) ++left;
    while (right > start && list_cpu[right * 5 + 4] <= pivot_score) --right;
    if (left <= right) {
      for (int i = 0; i < 5; ++i) {
        temp[i] = list_cpu[left * 5 + i];
      }
      for (int i = 0; i < 5; ++i) {
        list_cpu[left * 5 + i] = list_cpu[right * 5 + i];
      }
      for (int i = 0; i < 5; ++i) {
        list_cpu[right * 5 + i] = temp[i];
      }
      ++left;
      --right;
    }
  }

  if (right > start) {
    for (int i = 0; i < 5; ++i) {
      temp[i] = list_cpu[start * 5 + i];
    }
    for (int i = 0; i < 5; ++i) {
      list_cpu[start * 5 + i] = list_cpu[right * 5 + i];
    }
    for (int i = 0; i < 5; ++i) {
      list_cpu[right * 5 + i] = temp[i];
    }
  }

  if (start < right - 1) {
    sort_box(list_cpu, start, right - 1, num_top);
  }
  if (right + 1 < num_top && right + 1 < end) {
    sort_box(list_cpu, right + 1, end, num_top);
  }
}

// generate all candidate boxes with their scores
//   bottom: 1 x num_anchors x H x W tensor
//     bottom[0, k, h, w] = foreground score of anchor k at node (h, w)
//   d_anchor: num_anchors x 4 x H x W tensor
//     d_anchor[k, :, h, w] = gradient (dx, dy, d(log w), d(log h))
//                            of anchor k at center location (h, w)
//   num_anchors: number of anchors  (= # scales * # ratios)
//   anchors: num_anchors * 4 array,  (x1, y1, x2, y2) for each anchor
//   img_H, img_W: scaled image height & width
//   min_box_H, min_box_W: minimum box height & width
//   feat_stride: scaled image height (width) / bottom height (width)
//   proposals: num_proposals * 5 array
//     num_proposals = num_anchors * H * W
//     (x1, y1, x2, y2, score) for each proposal
#ifdef GPU
__global__
static
void enumerate_proposals_gpu(const real bottom4d[],
                             const real d_anchor4d[],
                             const real anchors[],
                             real proposals[],
                             const int num_anchors,
                             const int bottom_H, const int bottom_W,
                             const real img_H, const real img_W,
                             const real min_box_H, const real min_box_W,
                             const int feat_stride)
{
  const int bottom_area = bottom_H * bottom_W;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num_anchors * bottom_area) {
    const int h = index / num_anchors / bottom_W;
    const int w = (index / num_anchors) % bottom_W;
    const int k = index % num_anchors;
    const real x = w * feat_stride;
    const real y = h * feat_stride;
    const real* p_box = d_anchor4d + h * bottom_W + w;
    const real* p_score = bottom4d + h * bottom_W + w;

    const real dx = p_box[(k * 4 + 0) * bottom_area];
    const real dy = p_box[(k * 4 + 1) * bottom_area];
    const real d_log_w = p_box[(k * 4 + 2) * bottom_area];
    const real d_log_h = p_box[(k * 4 + 3) * bottom_area];

    proposals[index * 5 + 0] = x + anchors[k * 4 + 0];
    proposals[index * 5 + 1] = y + anchors[k * 4 + 1];
    proposals[index * 5 + 2] = x + anchors[k * 4 + 2];
    proposals[index * 5 + 3] = y + anchors[k * 4 + 3];

    proposals[index * 5 + 4]
        = transform_box(&proposals[index * 5],
                        dx, dy, d_log_w, d_log_h,
                        img_W, img_H, min_box_W, min_box_H)
          * p_score[k * bottom_area];
  }
}
#else
static
void enumerate_proposals_cpu(const real bottom4d[],
                             const real d_anchor4d[],
                             const real anchors[],
                             real proposals[],
                             const int num_anchors,
                             const int bottom_H, const int bottom_W,
                             const real img_H, const real img_W,
                             const real min_box_H, const real min_box_W,
                             const int feat_stride)
{
  const int bottom_area = bottom_H * bottom_W;
  for (int h = 0; h < bottom_H; ++h) {
    for (int w = 0; w < bottom_W; ++w) {
      const real x = w * feat_stride;
      const real y = h * feat_stride;
      const real* p_box = d_anchor4d + h * bottom_W + w;
      const real* p_score = bottom4d + h * bottom_W + w;
      for (int k = 0; k < num_anchors; ++k) {
        const real dx = p_box[(k * 4 + 0) * bottom_area];
        const real dy = p_box[(k * 4 + 1) * bottom_area];
        const real d_log_w = p_box[(k * 4 + 2) * bottom_area];
        const real d_log_h = p_box[(k * 4 + 3) * bottom_area];

        const int index = (h * bottom_W + w) * num_anchors + k;
        proposals[index * 5 + 0] = x + anchors[k * 4 + 0];
        proposals[index * 5 + 1] = y + anchors[k * 4 + 1];
        proposals[index * 5 + 2] = x + anchors[k * 4 + 2];
        proposals[index * 5 + 3] = y + anchors[k * 4 + 3];

        proposals[index * 5 + 4]
            = transform_box(&proposals[index * 5],
                            dx, dy, d_log_w, d_log_h,
                            img_W, img_H, min_box_W, min_box_H)
              * p_score[k * bottom_area];
      } // endfor k
    } // endfor w
  } // endfor h
}
#endif

// retrieve proposals that are determined to be kept as RoIs by NMS
//   proposals : "num_boxes x 5" array,  (x1, y1, x2, y2, score) for each box
//   num_rois: number of RoIs to be retrieved
//   index_roi: "num_rois x 1" array
//     index_roi[i]: index of i-th RoI in proposals
//   rois: "num_rois x 5" array,  (x1, y1, x2, y2, score) for each RoI
#ifdef GPU
__global__
static
void retrieve_rois_gpu(const real proposals[],
                       const int index_roi[],
                       real rois[],
                       const int num_rois)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num_rois) {
    const real* const proposals_index = proposals + index_roi[index] * 5;
    rois[index * 5 + 0] = proposals_index[0];
    rois[index * 5 + 1] = proposals_index[1];
    rois[index * 5 + 2] = proposals_index[2];
    rois[index * 5 + 3] = proposals_index[3];
    rois[index * 5 + 4] = proposals_index[4];
  }
}
#else
void retrieve_rois_cpu(const real proposals[],
                       const int index_roi[],
                       real rois[],
                       const int num_rois)
{
  for (int i = 0; i < num_rois; ++i) {
    const real* const proposals_index = proposals + index_roi[i] * 5;
    rois[i * 5 + 0] = proposals_index[0];
    rois[i * 5 + 1] = proposals_index[1];
    rois[i * 5 + 2] = proposals_index[2];
    rois[i * 5 + 3] = proposals_index[3];
    rois[i * 5 + 4] = proposals_index[4];
  }
}
#endif



// --------------------------------------------------------------------------
// layer operator code
// --------------------------------------------------------------------------

// auxiliary data structure for proposal operation
//   nms_aux_data: auxiliary data for NMS operation, (de)allocated by nms.cu
//   anchors: "num_anchors * 4" array of anchor boxes
//            (x1, y1, x2, y2) for each anchor
typedef struct ProposalAuxData_ {
  void* nms_aux_data;
  real* anchors;
} ProposalAuxData;

static
void malloc_proposal_aux_data(ProposalAuxData* const aux_data,
                              const LayerOption* const option,
                              long int* const p_space_cpu,
                              long int* const p_space)
{
  long int space_cpu = 0, space = 0;

  // auxiliary data for NMS
  malloc_nms_aux_data(&aux_data->nms_aux_data, option->pre_nms_topn,
                      &space_cpu, &space);

  // anchors
  {
    const int num_anchors = option->num_scales * option->num_ratios;
    real* const anchors = (real*)malloc(num_anchors * 4 * sizeof(real));
    generate_anchors(option->scales, option->ratios, anchors,
                     option->num_scales, option->num_ratios,
                     option->base_size);
    #ifdef GPU
    cudaMalloc(&aux_data->anchors, num_anchors * 4 * sizeof(real));
    cudaMemcpyAsync(aux_data->anchors, anchors,
                    num_anchors * 4 * sizeof(real),
                    cudaMemcpyHostToDevice);
    free(anchors);
    #else
    aux_data->anchors = anchors;
    #endif
    space += num_anchors * 4 * sizeof(real);
  }

  *p_space_cpu = space_cpu;
  *p_space = space;
}

static
void free_proposal_aux_data(ProposalAuxData* const aux_data)
{
  // auxiliary data for NMS
  free_nms_aux_data(aux_data->nms_aux_data);

  // anchors
  #ifdef GPU
  cudaFree(aux_data->anchors);
  #else
  free(aux_data->anchors);
  #endif

  memset(aux_data, 0, sizeof(ProposalAuxData));
}

// proposal: bottom -> top
//   bottom: (2 x num_anchors) x H x W tensor
//     bottom[0, k, h, w] = background score of anchor k at node (h, w)
//     bottom[1, k, h, w] = foreground score of anchor k at node (h, w)
//   d_anchor: (num_anchors x 4) x H x W tensor
//     d_anchor[k, :, h, w] = gradient (dx, dy, d(log w), d(log h))
//                            of anchor k at center location (h, w)
//   img_info: 6 x 1 tensor,  (img_H, img_W, scale_H, scale_W, raw_H, raw_W)
//     img_H, img_W: scaled image height & width
//     scale_H: height scale factor
//              img_H = raw image height * scale_H
//     scale_W: width scale factor
//              img_W = raw image width * scale_W
//     raw_H, raw_W: raw image height & width
//   top: num_RoIs x 5 tensor,  (x1, y1, x2, y2, score) of each RoI
//   aux_data: auxiliary data for proposal operation
//   temp_{cpu,gpu}_data: temporary space at {CPU, GPU} memory
static
void proposal_forward(const Tensor* const bottom4d,
                      const Tensor* const d_anchor4d,
                      const Tensor* const img_info1d,
                      Tensor* const top2d,
                      ProposalAuxData* const aux_data,
                      unsigned char temp_cpu_data[],
                      unsigned char temp_gpu_data[],
                      const LayerOption* const option)
{
  // number of anchors  (= number of scales * ratios)
  const int num_anchors = option->num_ratios * option->num_scales;

  // do forward-pass for each item in the batch
  const real* p_bottom_item = bottom4d->data;
  const real* p_d_anchor_item = d_anchor4d->data;
  real* p_top_item = top2d->data;
  int total_top_size = 0;

  // proposals: all box proposals with their scores
  //   "num_boxes x 5" array,  (x1, y1, x2, y2, score) for each box
  //   proposals_cpu: allocated at main memory
  //   proposals_gpu: allocated at GPU memory, only required in GPU mode
  // index_roi: indices of proposals to be retrieved as RoIs
  //   "num_rois x 1" array,  index_roi[i]: index of i-th RoI in proposals
  //   index_roi_cpu: allocated at main memory
  //   index_roi_gpu: allocated at GPU memory, only required in GPU mode
  // we divide temp space into two parts for index_roi and proposals
  //   index_roi = temp[0, ..., index_roi_size - 1]
  //   proposals = temp[index_roi_size, ..., ]
  //   index_roi_size <= option->post_nms_topn
  int* const index_roi_cpu = (int*)&temp_cpu_data[0];
  real* const proposals_cpu =
      (real*)&temp_cpu_data[option->post_nms_topn * sizeof(real)];
  #ifdef GPU
  int* const index_roi_gpu = (int*)&temp_gpu_data[0];
  real* const proposals_gpu =
      (real*)&temp_gpu_data[option->post_nms_topn * sizeof(real)];
  #endif

  #ifdef GPU
  real img_info_cpu[BATCH_SIZE * 6];
  const real* p_img_info_cpu = img_info_cpu;
  cudaMemcpyAsync(img_info_cpu, img_info1d->data,
                  get_data_size(img_info1d) * sizeof(real),
                  cudaMemcpyDeviceToHost);
  #else
  const real* p_img_info_cpu = img_info1d->data;
  #endif

  tick00 = clock();

  for (int n = 0; n < bottom4d->num_items; ++n) {
    // bottom shape: 2 x num_anchors x H x W
    const int bottom_H = bottom4d->shape[n][1];
    const int bottom_W = bottom4d->shape[n][2];
    // input image height & width
    const real img_H = p_img_info_cpu[0];
    const real img_W = p_img_info_cpu[1];
    // scale factor for height & width
    const real scale_H = p_img_info_cpu[2];
    const real scale_W = p_img_info_cpu[3];
    // minimum box width & height
    const real min_box_H = option->min_size * scale_H;
    const real min_box_W = option->min_size * scale_W;
    // number of all proposals = num_anchors * H * W
    const int num_proposals = num_anchors * bottom_H * bottom_W;

    tick0 = clock();
    // enumerate all proposals
    //   num_proposals = num_anchors * H * W
    //   (x1, y1, x2, y2, score) for each proposal
    // NOTE: for bottom, only foreground scores are passed
    #ifdef GPU
    {
      const int num_threads = num_proposals;
      const int threads_per_block = 512;
      const int num_blocks = DIV_THEN_CEIL(num_threads,  threads_per_block);
      enumerate_proposals_gpu<<<num_blocks, threads_per_block>>>(
          p_bottom_item + num_proposals,  p_d_anchor_item,
          aux_data->anchors,  proposals_gpu,  num_anchors,
          bottom_H,  bottom_W,  img_H,  img_W,  min_box_H,  min_box_W,
          option->feat_stride);
    }
    #else
    {
      enumerate_proposals_cpu(
          p_bottom_item + num_proposals,  p_d_anchor_item,
          aux_data->anchors,  proposals_cpu,  num_anchors,
          bottom_H,  bottom_W,  img_H,  img_W,  min_box_H,  min_box_W,
          option->feat_stride);
    }
    #endif
    tick1 = clock();
    a_time[0] = (float)(tick1 - tick0) / CLOCKS_PER_SEC;

    tick0 = clock();
    // choose candidates according to scores
    #ifdef GPU
    {
      cudaMemcpyAsync(proposals_cpu, proposals_gpu,
                      num_proposals * 5 * sizeof(real),
                      cudaMemcpyDeviceToHost);
      sort_box(proposals_cpu, 0, num_proposals - 1, option->pre_nms_topn);
      cudaMemcpyAsync(proposals_gpu, proposals_cpu,
                      num_proposals * 5 * sizeof(real),
                      cudaMemcpyHostToDevice);
    }
    #else
    {
      sort_box(proposals_cpu, 0, num_proposals - 1, option->pre_nms_topn);
    }
    #endif
    tick1 = clock();
    a_time[1] = (float)(tick1 - tick0) / CLOCKS_PER_SEC;

    tick0 = clock();
    // NMS & RoI retrieval
    {
      // NMS
      int num_rois = 0;
      {
        #ifdef GPU
        real* const p_proposals = proposals_gpu;
        #else
        real* const p_proposals = proposals_cpu;
        #endif
        nms(MIN(num_proposals,  option->pre_nms_topn),  p_proposals,
            aux_data->nms_aux_data,  &num_rois,  index_roi_cpu,  0,
            option->nms_thresh,  option->post_nms_topn,
            option->bbox_vote,  option->vote_thresh);
      }

      // RoI retrieval
      #ifdef GPU
      {
        const int num_threads = num_rois;
        const int threads_per_block = 128;
        const int num_blocks
            = DIV_THEN_CEIL(num_threads,  threads_per_block);

        cudaMemcpyAsync(index_roi_gpu, index_roi_cpu,
                        num_rois * sizeof(int),
                        cudaMemcpyHostToDevice);

        retrieve_rois_gpu<<<num_blocks, threads_per_block>>>(
            proposals_gpu,  index_roi_gpu,  p_top_item,  num_rois);
      }
      #else
      {
        retrieve_rois_cpu(
            proposals_cpu,  index_roi_cpu,  p_top_item,  num_rois);
      }
      #endif

      // set top shape: num_rois x 5,  (x1, y1, x2, y2, score) for each RoI
      top2d->shape[n][0] = num_rois;
      top2d->shape[n][1] = 5;
      top2d->start[n] = total_top_size;
      total_top_size += num_rois * 5;
    }
    tick1 = clock();
    a_time[2] = (float)(tick1 - tick0) / CLOCKS_PER_SEC;

    // locate next item
    {
      const int bottom_size = 2 * num_proposals;
      const int d_anchor_size = 4 * num_proposals;
      const int img_info_size = 6;
      const int top_size = 5 * top2d->shape[n][0];
      p_bottom_item += bottom_size;
      p_d_anchor_item += d_anchor_size;
      p_img_info_cpu += img_info_size;
      p_top_item += top_size;
    }
  } // endfor batch

  top2d->ndim = 2;
  top2d->num_items = bottom4d->num_items;

  tick1 = clock();
  a_time[3] = (float)(tick1 - tick00) / CLOCKS_PER_SEC;
  a_time[7] += (float)(tick1 - tick00) / CLOCKS_PER_SEC;

  print_tensor_info(top2d);
}



// --------------------------------------------------------------------------
// layer shape calculator code
// --------------------------------------------------------------------------
static
void proposal_shape(const Tensor* const bottom4d,
                    Tensor* const top2d,
                    long int* const p_temp_space,
                    const LayerOption* const option)
{
  int max_area = 0;

  // calculate shape for each item in the batch
  top2d->ndim = 2;
  top2d->num_items = bottom4d->num_items;
  for (int n = 0; n < bottom4d->num_items; ++n) {
    // calculate maximum area size for determining temporary space size
    const int bottom_H = bottom4d->shape[n][1];
    const int bottom_W = bottom4d->shape[n][2];
    const int bottom_area = bottom_H * bottom_W;
    max_area = MAX(max_area,  bottom_area);

    // top shape <= post_nms_topn x 5
    //   exact row size will be determined after forward-pass
    top2d->shape[n][0] = option->post_nms_topn;
    top2d->shape[n][1] = 5;
    top2d->start[n] = top2d->shape[n][0] * top2d->shape[n][1];
  }

  // temporary space size
  {
    const int num_anchors = option->num_ratios * option->num_scales;
    const int proposals_size = num_anchors * max_area * 5;
    const int index_roi_size = option->post_nms_topn;

    *p_temp_space = proposals_size * sizeof(real)
                    + index_roi_size * sizeof(int);
  }
}



// --------------------------------------------------------------------------
// API code
// --------------------------------------------------------------------------

void forward_proposal_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;

  proposal_forward(get_bottom(layer, 0), get_bottom(layer, 1),
                   get_bottom(layer, 2),
                   get_top(layer, 0),
                   (ProposalAuxData*)layer->aux_data,
                   (unsigned char*)net->temp_cpu_data,
                   (unsigned char*)net->temp_data,
                   &layer->option);

  #ifdef DEBUG
  {
    printf("%s:  ", layer->name);
    for (int i = 0; i < 8; ++i) {
      printf("%4.2f\t", a_time[i] * 1000);
    }
    printf("\n");
  }
  #endif
}

void shape_proposal_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;
  long int temp_space;

  proposal_shape(get_bottom(layer, 0), get_top(layer, 0),
                 &temp_space, &layer->option);

  update_temp_space(net, temp_space);
}

void malloc_proposal_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;
  long int space_cpu, space;

  layer->aux_data = (void*)malloc(sizeof(ProposalAuxData));

  malloc_proposal_aux_data((ProposalAuxData*)layer->aux_data,
                           &layer->option,
                           &space_cpu, &space);

  net->space_cpu += space_cpu + sizeof(ProposalAuxData);
  net->space += space;

  #ifdef DEBUG
  {
    #ifdef GPU
    printf("%s: Memory allocated, CPU %ld byte and GPU %ld byte\n",
           layer->name, space_cpu, space);
    #else
    printf("%s: Memory allocated, %ld byte\n",
           layer->name, space_cpu + space);
    #endif
  }
  #endif
}

void free_proposal_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;

  free_proposal_aux_data((ProposalAuxData*)layer->aux_data);
  free(layer->aux_data);
}
