#include "layer.h"
#include <stdlib.h>
#include <math.h>

#ifdef GPU
#include "cuda_settings.h"
#endif

// "IoU = intersection area / union area" of two boxes A, B
//   A, B: 4-dim array (x1, y1, x2, y2)
#ifdef GPU
__device__
#endif
inline real iou(const real* const A, const real* const B)
{
  // overlapped region (= box)
  const real x1 = MAX(A[0], B[0]);
  const real y1 = MAX(A[1], B[1]);
  const real x2 = MIN(A[2], B[2]);
  const real y2 = MIN(A[3], B[3]);

  // intersection area
  const real width = MAX(0.0f,  x2 - x1 + 1.0f);
  const real height = MAX(0.0f,  y2 - y1 + 1.0f);
  const real area = width * height;

  // area of A, B
  const real A_area = (A[2] - A[0] + 1.0f) * (A[3] - A[1] + 1.0f);
  const real B_area = (B[2] - B[0] + 1.0f) * (B[3] - B[1] + 1.0f);

  // IoU
  return area / (A_area + B_area - area);
}

// the whole 2-dim computations "num_boxes x num_boxes" is done by
// divide-and-conquer computations:
//   each GPU block performs "64 x 64" computations,
//   and each "1 x 64" result is saved into a 64-bit mask
typedef unsigned long long uint64;
#define NMS_BLOCK_SIZE 64

#ifdef GPU
__global__
void nms_mask_gpu(const real* const boxes,
                  uint64* const mask,
                  const int num_boxes, const real nms_thresh)
{
  // block region
  //   j = j_start + { 0, ..., dj_end - 1 }
  //   i = i_start + { 0, ..., di_end - 1 }
  const int i_start = blockIdx.x * NMS_BLOCK_SIZE;
  const int di_end = MIN(num_boxes - i_start,  NMS_BLOCK_SIZE);
  const int j_start = blockIdx.y * NMS_BLOCK_SIZE;
  const int dj_end = MIN(num_boxes - j_start,  NMS_BLOCK_SIZE);

  // copy all i-th boxes to GPU cache
  //   i = i_start + { 0, ..., di_end - 1 }
  __shared__ real boxes_i[NMS_BLOCK_SIZE * 5];
  {
    const int di = threadIdx.x;
    if (di < di_end) {
      boxes_i[di * 5 + 0] = boxes[(i_start + di) * 5 + 0];
      boxes_i[di * 5 + 1] = boxes[(i_start + di) * 5 + 1];
      boxes_i[di * 5 + 2] = boxes[(i_start + di) * 5 + 2];
      boxes_i[di * 5 + 3] = boxes[(i_start + di) * 5 + 3];
      boxes_i[di * 5 + 4] = boxes[(i_start + di) * 5 + 4];
    }
  }
  __syncthreads();

  // given j = j_start + dj,
  //   check whether box i is significantly overlapped with box j
  //   (i.e., IoU(box j, box i) > threshold)
  //   for all i = i_start + { 0, ..., di_end - 1 } except for i == j
  {
    const int dj = threadIdx.x;
    if (dj < dj_end) {
      // box j
      const real* const box_j = boxes + (j_start + dj) * 5;

      // mask for significant overlap
      //   if IoU(box j, box i) > threshold,  di-th bit = 1
      uint64 mask_j = 0;

      // check for all i = i_start + { 0, ..., di_end - 1 }
      // except for i == j
      const int di_start = (i_start == j_start) ? (dj + 1) : 0;
      for (int di = di_start; di < di_end; ++di) {
        // box i
        const real* const box_i = boxes_i + di * 5;

        // if IoU(box j, box i) > threshold,  di-th bit = 1
        if (iou(box_j, box_i) > nms_thresh) {
          mask_j |= 1ULL << di;
        }
      }

      // mask: "num_boxes x num_blocks" array
      //   for mask[j][bi], "di-th bit = 1" means:
      //     box j is significantly overlapped with box i = i_start + di,
      //     where i_start = bi * block_size
      {
        const int num_blocks = DIV_THEN_CEIL(num_boxes, NMS_BLOCK_SIZE);
        const int bi = blockIdx.x;
        mask[(j_start + dj) * num_blocks + bi] = mask_j;
      }
    } // endif dj < dj_end
  }
}
#else
void nms_mask_cpu(const real* const boxes,
                  uint64* const mask,
                  const int num_boxes, const real nms_thresh)
{
  // number of blocks along each dimension
  const int num_blocks = DIV_THEN_CEIL(num_boxes, NMS_BLOCK_SIZE);

  // the whole 2-dim computations "num_boxes x num_boxes" is done by
  // sweeping all "64 x 64"-sized blocks
  for (int j_start = 0; j_start < num_boxes; j_start += NMS_BLOCK_SIZE) {
    for (int i_start = 0; i_start < num_boxes; i_start += NMS_BLOCK_SIZE) {
      // block region
      //   j = j_start + { 0, ..., dj_end - 1 }
      //   i = i_start + { 0, ..., di_end - 1 }
      const int di_end = MIN(num_boxes - i_start,  NMS_BLOCK_SIZE);
      const int dj_end = MIN(num_boxes - j_start,  NMS_BLOCK_SIZE);

      // check whether box i is significantly overlapped with box j
      // for all j = j_start + { 0, ..., dj_end - 1 },
      //         i = i_start + { 0, ..., di_end - 1 },
      // except for i == j
      for (int dj = 0; dj < dj_end; ++dj) {
        // box j & overlap mask
        const real* const box_j = boxes + (j_start + dj) * 5;
        uint64 mask_j = 0;

        // check for all i = i_start + { 0, ..., di_end - 1 }
        // except for i == j
        const int di_start = (i_start == j_start) ? (dj + 1) : 0;
        for (int di = di_start; di < di_end; ++di) {
          // box i
          const real* const box_i = boxes + (i_start + di) * 5;

          // if IoU(box j, box i) > threshold,  di-th bit = 1
          if (iou(box_j, box_i) > nms_thresh) {
            mask_j |= 1ULL << di;
          }
        }

        // mask: "num_boxes x num_blocks" array
        //   for mask[j][bi], "di-th bit = 1" means:
        //     box j is significantly overlapped with box i = i_start + di,
        //     where i_start = bi * block_size
        {
          const int bi = i_start / NMS_BLOCK_SIZE;
          mask[(j_start + dj) * num_blocks + bi] = mask_j;
        }
      } // endfor dj
    } // endfor j_start
  } // endfor i_start
}
#endif

// given box proposals (sorted in descending order of their scores),
// discard a box if it is significantly overlapped with
// one or more previous (= scored higher) boxes
//   num_boxes: number of box proposals given
//   boxes: "num_boxes x 5" array (x1, y1, x2, y2, score)
//          sorted in descending order of scores
//   num_out: number of remaining boxes
//   keep_out: "num_out x 1" array
//             indices of remaining boxes
//   nms_thresh: threshold for determining "significant overlap"
//               if "intersection area / union area > nms_thresh",
//               two boxes are thought of as significantly overlapped
void nms(const int num_boxes, const real* const boxes,
         int* const num_out, int* const keep_out,
         const real nms_thresh)
{
  const int num_blocks = DIV_THEN_CEIL(num_boxes, NMS_BLOCK_SIZE);
  uint64* const mask
      = (uint64*)malloc(num_boxes * num_blocks * sizeof(uint64));

  #ifdef GPU
  {
    uint64* mask_dev;
    real* boxes_dev;
    const dim3 blocks(num_blocks, num_blocks);

    // GPU memory allocation & copy box data
    CUDA_CHECK(cudaMalloc(&boxes_dev, num_boxes * 5 * sizeof(real)));
    CUDA_CHECK(cudaMemcpy(boxes_dev, boxes, num_boxes * 5 * sizeof(real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&mask_dev,
                          num_boxes * num_blocks * sizeof(uint64)));

    // find all significantly-overlapped pairs of boxes
    nms_mask_gpu<<<blocks, NMS_BLOCK_SIZE>>>(
        boxes_dev, mask_dev, num_boxes, nms_thresh);

    // copy mask data to main memory
    CUDA_CHECK(cudaMemcpy(mask, mask_dev,
                          sizeof(uint64) * num_boxes * num_blocks,
                          cudaMemcpyDeviceToHost));

    // GPU memory deallocation
    CUDA_CHECK(cudaFree(boxes_dev));
    CUDA_CHECK(cudaFree(mask_dev));
  }
  #else
  {
    // find all significantly-overlapped pairs of boxes
    nms_mask_cpu(boxes, mask, num_boxes, nms_thresh);
  }
  #endif

  // discard i-th box if it is significantly overlapped with
  // one or more previous (= scored higher) boxes
  {
    int num_to_keep = 0;
    uint64* const remv = (uint64*)calloc(num_blocks, sizeof(uint64));

    for (int i = 0; i < num_boxes; ++i) {
      const int nblock = i / NMS_BLOCK_SIZE;
      const int inblock = i % NMS_BLOCK_SIZE;

      if (!(remv[nblock] & (1ULL << inblock))) {
        keep_out[num_to_keep++] = i;
        uint64* p = mask + i * num_blocks;
        for (int j = nblock; j < num_blocks; ++j) {
          remv[j] |= p[j];
        }
      }
    }
    *num_out = num_to_keep;

    free(remv);
  }

  free(mask);
}

typedef struct BoundingBox_
{
  real x1, y1, x2, y2;
  real score;
} BoundingBox;

int transform_box(BoundingBox* const box,
                  const real dx, const real dy,
                  const real dw, const real dh,
                  const real im_w, const real im_h,
                  const real min_w, const real min_h)
{
  const real w = box->x2 - box->x1 + 1.0f;
  const real h = box->y2 - box->y1 + 1.0f;
  const real ctr_x = box->x1 + 0.5f * w;
  const real ctr_y = box->y1 + 0.5f * h;

  const real pred_ctr_x = dx * w + ctr_x;
  const real pred_ctr_y = dy * h + ctr_y;
  const real pred_w = exp(dw) * w;
  const real pred_h = exp(dh) * h;

  box->x1 = pred_ctr_x - 0.5f * pred_w;
  box->y1 = pred_ctr_y - 0.5f * pred_h;
  box->x2 = pred_ctr_x + 0.5f * pred_w;
  box->y2 = pred_ctr_y + 0.5f * pred_h;

  box->x1 = MAX(0.0f,  MIN(box->x1,  im_w - 1.0f));
  box->y1 = MAX(0.0f,  MIN(box->y1,  im_h - 1.0f));
  box->x2 = MAX(0.0f,  MIN(box->x2,  im_w - 1.0f));
  box->y2 = MAX(0.0f,  MIN(box->y2,  im_h - 1.0f));

  const real box_w = box->x2 - box->x1 + 1.0f;
  const real box_h = box->y2 - box->y1 + 1.0f;

  if (box_w >= min_w && box_h >= min_h) return 1;
  return 0;
}

#define MAX_NUM_RATIO_SCALE 10
#define MAX_DATA_WIDTH 80
#define MAX_DATA_HEIGHT 80
#define MAX_NUM_PROPOSAL 6000

void generate_anchors(real* const anchors,
                      const ProposalOption* const option)
{
  const real base_area = option->base_size * option->base_size;
  const real ctr = 0.5f * (option->base_size - 1.0f);
  real wr[MAX_NUM_RATIO_SCALE];
  real hr[MAX_NUM_RATIO_SCALE];
  for (int i = 0; i < option->num_ratios; ++i) {
    wr[i] = ROUND(sqrt(base_area / option->ratios[i]));
    hr[i] = ROUND(wr[i] * option->ratios[i]);
  }

  // anchor generation
  {
    real* p_anchors = anchors;
    for (int c = 0; c < option->num_concats; ++c) {
      for (int i = 0; i < option->num_ratios; ++i) {
        for (int j = 0; j < option->num_scales; ++j) {
          const real ws = 0.5f * (wr[i] * option->scales[j] - 1.0f);
          const real hs = 0.5f * (hr[i] * option->scales[j] - 1.0f);
          p_anchors[0] = ctr - ws;
          p_anchors[1] = ctr - hs;
          p_anchors[2] = ctr + ws;
          p_anchors[3] = ctr + hs;
          p_anchors += 4;
        } // endfor j
      } // endfor i
    } // endfor c
  }
}

// quick-sort a list of boxes in descending order of their scores
//   if num_top <= end,  only top-k results are guaranteed to be sorted
//   (for efficient computation)
void sort_box(BoundingBox* const list, const int start, const int end,
              const int num_top)
{
  const real pivot_score = list[start].score;
  int left = start + 1, right = end;
  BoundingBox temp;
  while (left <= right) {
    while (left <= end && list[left].score >= pivot_score) ++left;
    while (right > start && list[right].score <= pivot_score) --right;
    if (left <= right) {
      temp = list[left];
      list[left] = list[right];
      list[right] = temp;
      ++left;
      --right;
    }
  }
  if (right > start) {
    temp = list[right];
    list[right] = list[start];
    list[start] = temp;
  }
  if (start < right - 1) {
    sort_box(list, start, right - 1, num_top);
  }
  if (right + 1 < num_top && right + 1 < end) {
    sort_box(list, right + 1, end, num_top);
  }
}

void proposal_forward(const Tensor* const bottom4d,
                      const Tensor* const pred_box4d,
                      const Tensor* const img_info1d,
                      Tensor* const top2d,
                      const real* const anchors,
                      const ProposalOption* const option)
{
  BoundingBox* const proposals
      = (BoundingBox*)malloc(MAX_NUM_RATIO_SCALE * MAX_NUM_RATIO_SCALE *
                             MAX_DATA_WIDTH * MAX_DATA_HEIGHT *
                             sizeof(BoundingBox));
  real* const sorted_dets
      = (real*)malloc(MAX_NUM_PROPOSAL * 5 * sizeof(real));
  int* const keep = (int*)malloc(MAX_NUM_PROPOSAL * sizeof(int));

  // bottom4d: N x 2 x num_anchors x H x W
  // pred_box4d: N x num_anchors x 4 x H x W
  // img_info1d: N x 4
  // top2d: N x num_rois x 4
  const real* p_bottom_data = bottom4d->data;
  const real* p_pred_box_data = pred_box4d->data;
  const real* p_img_info = img_info1d->data;
  real* p_top_data = top2d->data;
  const int num_anchors
      = option->num_concats * option->num_ratios * option->num_scales;
  for (int n = 0; n < bottom4d->num_items; ++n) {
    const int H = bottom4d->shape[n][2];
    const int W = bottom4d->shape[n][3];
    const int HW = H * W;
    const real im_w = p_img_info[1];
    const real im_h = p_img_info[0];
    const real min_w = option->min_size * p_img_info[2];
    const real min_h = option->min_size * p_img_info[3];

    // enumerate all proposals
    int num_proposals = 0;
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        const real x = w * option->feat_stride;
        const real y = h * option->feat_stride;
        const real* p_box = &p_pred_box_data[h * W + w];
        const real* p_score = &p_bottom_data[num_anchors * HW + h * W + w];
        for (int k = 0; k < num_anchors; ++k) {
          const real dx = p_box[(k * 4 + 0) * HW];
          const real dy = p_box[(k * 4 + 1) * HW];
          const real dw = p_box[(k * 4 + 2) * HW];
          const real dh = p_box[(k * 4 + 3) * HW];
          proposals[num_proposals].x1 = x + anchors[k * 4 + 0];
          proposals[num_proposals].y1 = y + anchors[k * 4 + 1];
          proposals[num_proposals].x2 = x + anchors[k * 4 + 2];
          proposals[num_proposals].y2 = y + anchors[k * 4 + 3];
          proposals[num_proposals].score = p_score[k * HW];
          {
            const int box_created = transform_box(&proposals[num_proposals],
                                                  dx, dy, dw, dh,
                                                  im_w, im_h, min_w, min_h);
            if (box_created) ++num_proposals;
          }
        } // endfor k
      } // endfor w
    } // endfor h

    // choose candidates according to scores
    sort_box(proposals, 0, num_proposals - 1, option->pre_nms_topn);
    if (num_proposals > option->pre_nms_topn) {
      num_proposals = option->pre_nms_topn;
    }
    for (int i = 0; i < num_proposals; ++i) {
      sorted_dets[i * 5 + 0] = proposals[i].x1;
      sorted_dets[i * 5 + 1] = proposals[i].y1;
      sorted_dets[i * 5 + 2] = proposals[i].x2;
      sorted_dets[i * 5 + 3] = proposals[i].y2;
      sorted_dets[i * 5 + 4] = proposals[i].score;
    }

    // NMS & RoI retrieval
    {
      int num_rois = 0;
      nms(num_proposals, sorted_dets, &num_rois, keep, option->nms_thresh);

      if (num_rois > option->post_nms_topn) {
        num_rois = option->post_nms_topn;
      }
      top2d->shape[n][0] = num_rois;
      top2d->shape[n][1] = 4;
      for (int i = 0; i < num_rois; ++i) {
        p_top_data[i * 4 + 0] = proposals[keep[i]].x1;
        p_top_data[i * 4 + 1] = proposals[keep[i]].y1;
        p_top_data[i * 4 + 2] = proposals[keep[i]].x2;
        p_top_data[i * 4 + 3] = proposals[keep[i]].y2;
      }
    }

    // locate next item
    p_top_data += 4 * top2d->shape[n][0];
    p_bottom_data += 2 * num_anchors * HW;
    p_pred_box_data += 4 * num_anchors * HW;
    p_img_info += 4;
  } // endfor num_items

  top2d->ndim = 2;
  top2d->num_items = bottom4d->num_items;

  free(proposals);
  free(sorted_dets);
  free(keep);
}

// test code
#ifdef TEST
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  // variable declaration & memory allocation
  Tensor score, bbox, im_info, roi;
  real score_data[150*36*46], bbox_data[300*36*46], im_info_data[4];
  real roi_data[300*4], roi_true_data[300*4];
  real anchors[100];
  real scales[5] = {3, 6, 9, 16, 32};
  real ratios[5] = {0.5, 0.666, 1.0, 1.5, 2.0};
  int num_anchors;
  ProposalOption option;

  // set option
  {
    option.num_concats = 3;
    option.base_size = 16;
    option.feat_stride = 16;
    option.pre_nms_topn = 6000;
    option.post_nms_topn = 300;
    option.nms_thresh = 0.7;
    option.min_size = 16;
    option.scales = &scales[0];
    option.ratios = &ratios[0];
    option.num_scales = 5;
    option.num_ratios = 5;
  }

  // generate anchors
  {
    generate_anchors(anchors, &option);
    num_anchors = option.num_concats * option.num_scales * option.num_ratios;
  }

  // set data shapes
  {
    score.ndim = 4; score.num_items = 1; score.data = &score_data[0];
    for (int i = 0; i < score.num_items; ++i) {
      score.shape[i][0] = 2;
      score.shape[i][1] = num_anchors;
      score.shape[i][2] = 36;
      score.shape[i][3] = 46;
    }

    bbox.ndim = 4; bbox.num_items = score.num_items; bbox.data = &bbox_data[0];
    for (int i = 0; i < bbox.num_items; ++i) {
      bbox.shape[i][0] = num_anchors;
      bbox.shape[i][1] = 4;
      bbox.shape[i][2] = score.shape[i][2];
      bbox.shape[i][3] = score.shape[i][3];
    }

    im_info.ndim = 1; im_info.num_items = score.num_items; im_info.data = &im_info_data[0];
    for (int i = 0; i < im_info.num_items; ++i) {
      im_info.shape[i][0] = 4;
    }

    roi.ndim = 2; roi.num_items = score.num_items; roi.data = &roi_data[0];
    for (int i = 0; i < roi.num_items; ++i) {
      roi.shape[i][0] = option.post_nms_topn;
      roi.shape[i][1] = 4;
    }
  }

  // load data
  {
    FILE* fp;

    fp = fopen("../data/temp/proposal_bottom0.txt", "r");
    for (int i = 0; i < flatten_size(&score); ++i) {
      if (fscanf(fp, "%f", &score_data[i]) <= 0) {
        printf("Error occurred while reading proposal_bottom0[%d]\n", i);
      }
    }
    fclose(fp);

    fp = fopen("../data/temp/proposal_bottom1.txt", "r");
    for (int i = 0; i < flatten_size(&bbox); ++i)
      if (fscanf(fp, "%f", &bbox_data[i]) <= 0) {
        printf("Error occurred while reading proposal_bottom1[%d]\n", i);
      }
    fclose(fp);

    fp = fopen("../data/temp/proposal_bottom2.txt", "r");
    for (int i = 0; i < flatten_size(&im_info); ++i)
      if (fscanf(fp, "%f", &im_info_data[i]) <= 0) {
        printf("Error occurred while reading proposal_bottom2[%d]\n", i);
      }
    fclose(fp);

    fp = fopen("../data/temp/proposal_top0.txt", "r");
    for (int i = 0; i < flatten_size(&roi); ++i)
      if (fscanf(fp, "%f", &roi_true_data[i]) <= 0) {
        printf("Error occurred while reading proposal_top0[%d]\n", i);
      }
    fclose(fp);
  }

  // CUDA initialization
  #ifdef GPU
  {
    printf("set device\n");
    CUDA_CHECK(cudaSetDevice(0));
  }
  #endif

  // do forward operation
  {
    printf("do forward\n");
    proposal_forward(&score, &bbox, &im_info, &roi, anchors, &option);
  }

  // verify results
  {
    const int roi_size = flatten_size(&roi);
    int i = 0;
    int i_true = 0;
    for (; i < roi_size && i_true < roi_size; i += 4, i_true += 4) {
      real diff = 0.0f;
      for (int di = 0; di < 4; ++di) {
        diff += ABS(roi_data[i + di] - roi_true_data[i_true + di]) /
                (1e-10f + MIN(roi_data[i + di], roi_true_data[i_true + di]));
      }
      if (diff > 1e-3f) {
        real diff1 = 0.0f;
        for (int di = 0; di < 4; ++di) {
          diff1 += ABS(roi_data[i + di] - roi_true_data[i_true + 4 + di]) /
            (1e-10f + MIN(roi_data[i + di], roi_true_data[i_true + 4 + di]));
        }
        if (diff1 < 1e-3f) {
          printf("[Missed] RoI_true[%d] = %.2f %.2f %.2f %.2f\n", i_true / 4,
                 roi_true_data[i_true + 0], roi_true_data[i_true + 1],
                 roi_true_data[i_true + 2], roi_true_data[i_true + 3]);
          i_true += 4;
          continue;
        }
        real diff2 = 0.0f;
        for (int di = 0; di < 4; ++di) {
          diff1 += ABS(roi_data[i + 4 + di] - roi_true_data[i_true + di]) /
            (1e-10f + MIN(roi_data[i + 4 + di], roi_true_data[i_true + di]));
        }
        if (diff2 < 1e-3f) {
          printf("[False box] RoI[%d] = %.2f %.2f %.2f %.2f\n",
                 i / 4, roi_data[i + 0], roi_data[i + 1],
                 roi_data[i + 2], roi_data[i + 3]);
          i += 4;
          continue;
        }
        printf("RoI[%d] = %.2f %.2f %.2f %.2f  ",
               i / 4, roi_data[i + 0], roi_data[i + 1],
               roi_data[i + 2], roi_data[i + 3]);
        printf("RoI_true[%d] = %.2f %.2f %.2f %.2f\n",
               i / 4, roi_true_data[i + 0], roi_true_data[i + 1],
               roi_true_data[i + 2], roi_true_data[i + 3]);
      }
    }
    for (; i < roi_size; i += 4) {
      printf("[False box] RoI[%d] = %.2f %.2f %.2f %.2f\n",
             i / 4, roi_data[i + 0], roi_data[i + 1],
             roi_data[i + 2], roi_data[i + 3]);
    }
    for (; i_true < roi_size; i_true += 4) {
      printf("[Missed] RoI_true[%d] = %.2f %.2f %.2f %.2f\n", i_true / 4,
             roi_true_data[i_true + 0], roi_true_data[i_true + 1],
             roi_true_data[i_true + 2], roi_true_data[i_true + 3]);
    }
  }

  return 0;
}
#endif // endifdef TEST
