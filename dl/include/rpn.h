#ifndef PVA_DL_RPN_FUNC_H
#define PVA_DL_RPN_FUNC_H

#include "common.h"

// --------------------------------------------------------------------------
// functions related to region proposal operators
//   iou
//   soft_box
//   nms
// --------------------------------------------------------------------------

// "IoU = intersection area / union area" of two boxes A, B
//   A, B: 4-dim array (x1, y1, x2, y2)
real iou(const real A[], const real B[]);

// quick-sort a list of boxes in descending order of their scores (CPU)
//   list_cpu: num_boxes x 5 array,  (x1, y1, x2, y2, score) for each box
//             located at main memory
//   if num_top <= end,  only top-k results are guaranteed to be sorted
//   (for efficient computation)
void sort_box(real list_cpu[], const int start, const int end,
              const int num_top);

// given box proposals (sorted in descending order of their scores),
// discard a box if it is significantly overlapped with
// one or more previous (= scored higher) boxes
//   num_boxes: number of box proposals given
//   boxes: "num_boxes x 5" array (x1, y1, x2, y2, score)
//          sorted in descending order of scores
//   aux_data: auxiliary data for NMS operation
//   num_out: number of remaining boxes
//   index_out_cpu: "num_out x 1" array
//                  indices of remaining boxes
//                  allocated at main memory
//   base_index: a constant added to index_out_cpu,  usually 0
//               index_out_cpu[i] = base_index + actual index in boxes
//   nms_thresh: threshold for determining "significant overlap"
//               if "intersection area / union area > nms_thresh",
//               two boxes are thought of as significantly overlapped
//   bbox_vote: whether bounding-box voting is used (= 1) or not (= 0)
//   vote_thresh: threshold for selecting overlapped boxes
//                which are participated in bounding-box voting
void nms(const int num_boxes, real boxes[], void* const aux_data,
         int* const num_out, int index_out_cpu[],
         const int base_index, const real nms_thresh, const int max_num_out,
         const int bbox_vote, const real vote_thresh);

// NMS auxiliary data initializer
void malloc_nms_aux_data(void** const p_aux_data,
                         int num_boxes,
                         long int* const p_space_cpu,
                         long int* const p_space_gpu);

// NMS auxiliary data finalizer
void free_nms_aux_data(void* const aux_data);

#endif // end PVA_DL_RPN_FUNC_H
