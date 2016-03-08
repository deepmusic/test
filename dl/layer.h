#ifndef PVA_DL_LAYER_H
#define PVA_DL_LAYER_H

// --------------------------------------------------------------------------
// include cuda & blas library
// --------------------------------------------------------------------------

#ifdef GPU
  #include "cuda_settings.h"
#else
  #include <cblas.h>
#endif

#include <stdlib.h>



// --------------------------------------------------------------------------
// simple math operations
// --------------------------------------------------------------------------

#define ABS(x)  ((x) > 0 ? (x) : (-(x)))
#define DIV_THEN_CEIL(x, y)  (((x) + (y) - 1) / (y))
#define ROUND(x)  ((int)((x) + 0.5f))

#ifdef GPU
  #define MIN(x, y)  min(x, y)
  #define MAX(x, y)  max(x, y)
#else
  #define MIN(x, y)  ((x) < (y) ? (x) : (y))
  #define MAX(x, y)  ((x) > (y) ? (x) : (y))
#endif



// --------------------------------------------------------------------------
// tensor data structure & some functions
//   struct Tensor
//   flatten_size
//   load_data
// --------------------------------------------------------------------------

typedef float real;
#define g_max_num_items 128
#define g_max_ndim 5

typedef struct Tensor_
{
  int num_items;
  int ndim;
  int shape[g_max_num_items][g_max_ndim];
  int start[g_max_num_items];
  real* data;
} Tensor;

// total number of elements in a tensor
int flatten_size(const Tensor* const tensor);

// load binary data from file
//   if "data" is NULL, allocate memory & load data & return pointer
//   otherwise, load data to "data"
real* load_data(const char* const filename,
                int* const ndim,
                int* const shape,
                real* data);



// --------------------------------------------------------------------------
// convolution & deconvolution
//   struct ConvOption
//   conv_forward
//   conv_shape
//   deconv_forward
//   deconv_shape
// --------------------------------------------------------------------------

typedef struct ConvOption_
{
  int num_groups;
  int out_channels;
  int kernel_h, kernel_w;
  int pad_h, pad_w;
  int stride_h, stride_w;
  int bias;
  void* handle;
} ConvOption;

// convolution: bottom -> top
//   G: number of groups
//   bottom: (G * C) x H x W
//   top: (G * C') x H' x W'
//   weight: G x C' x C x kernel_h x kernel_w
//   bias: (G * C') x 1
//   temp: (G * C * kernel_h * kernel_w) x (H' * W') array
//   const: 1 x (H' * W') array,  const[i] = 1 for all i
void conv_forward(const Tensor* const bottom3d,
                  Tensor* const top3d,
                  const Tensor* const weight5d,
                  const Tensor* const bias1d,
                  real* const temp_data,
                  const real* const const_data,
                  const ConvOption* const option);

void conv_shape(const Tensor* const bottom3d,
                Tensor* const top3d,
                Tensor* const weight5d,
                Tensor* const bias1d,
                int* const temp_size,
                int* const const_size,
                const ConvOption* const option);

// deconvolution: bottom -> top
//   G: number of groups
//   bottom: (G * C') x H' x W'
//   top: (G * C) x H x W
//   weight: G x C' x C x kernel_h x kernel_w
//   bias: (G * C) x 1
//   temp: (G * C * kernel_h * kernel_w) x (H' * W') array
//   const: 1 x (H * W) array,  const[i] = 1 for all i
void deconv_forward(const Tensor* const bottom3d,
                    Tensor* const top3d,
                    const Tensor* const weight5d,
                    const Tensor* const bias1d,
                    real* const temp_data,
                    const real* const const_data,
                    const ConvOption* const option);

void deconv_shape(const Tensor* const bottom3d,
                  Tensor* const top3d,
                  Tensor* const weight5d,
                  Tensor* const bias1d,
                  int* const temp_size,
                  int* const const_size,
                  const ConvOption* const option);



// --------------------------------------------------------------------------
// fully-connected
//   struct FCOption
//   fc_forward
//   fc_shape
// --------------------------------------------------------------------------

typedef struct FCOption_
{
  int out_channels;
  int bias;
  void* handle;
} FCOption;

// fully-connected: bottom -> top
//   bottom: N x D (N items of D-dim array)
//   top: N x D' (N items of D-dim array)
//   weight: D' x D
//   bias: 1 x D'
//   const: N-dim array,  const[i] = 1 for all i
void fc_forward(const Tensor* const bottom2d,
                Tensor* const top2d,
                const Tensor* const weight2d,
                const Tensor* const bias1d,
                const real* const const_data,
                const FCOption* const option);

void fc_shape(const Tensor* const bottom2d,
              Tensor* const top2d,
              Tensor* const weight2d,
              Tensor* const bias1d,
              int* const const_size,
              const FCOption* const option);



// --------------------------------------------------------------------------
// pooling
//   struct PoolOption
//   pool_forward
//   pool_shape
// --------------------------------------------------------------------------

typedef struct PoolOption_
{
  int kernel_h, kernel_w;
  int pad_h, pad_w;
  int stride_h, stride_w;
} PoolOption;

// max-pooling: bottom -> top
//   bottom: C x H x W
//   top: C x H' x W'
//   argmax: C x H' x W' array
void pool_forward(const Tensor* const bottom3d,
                  Tensor* const top3d,
                  int* const argmax_data,
                  const PoolOption* const option);

void pool_shape(const Tensor* const bottom3d,
                Tensor* const top3d,
                int* const argmax_size,
                const PoolOption* const option);



// --------------------------------------------------------------------------
// RoI pooling
//   struct ROIPoolOption
//   roipool_forward
//   roipool_shape
// --------------------------------------------------------------------------

typedef struct ROIPoolOption_
{
  int pooled_height;
  int pooled_width;
  real spatial_scale;
} ROIPoolOption;

// RoI pooling: bottom -> top
//   bottom: C x H x W
//   roi: R x 4
//   top: R x C x H' x W'
//   argmax: R * C * H' * W' array
void roipool_forward(const Tensor* const bottom3d,
                     const Tensor* const roi2d,
                     Tensor* const top4d,
                     int* const argmax_data,
                     const ROIPoolOption* option);

void roipool_shape(const Tensor* const bottom3d,
                   const Tensor* const roi2d,
                   Tensor* const top4d,
                   int* const argmax_size,
                   const ROIPoolOption* option);



// --------------------------------------------------------------------------
// ReLU transform
//   struct ReluOption
//   relu_forward
//   relu_forward_inplace
//   relu_shape
// --------------------------------------------------------------------------

typedef struct ReluOption_
{
  real negative_slope;
} ReluOption;

// (soft-)ReLU transform: bottom -> top
//   data size: total number of nodes (N * C * H * W or something)
//   if option->negative_slope = 0, perform ReLU
//                             > 0, perform soft ReLU
void relu_forward(const Tensor* const bottom,
                  Tensor* const top,
                  const ReluOption* const option);

// in-place (soft-)ReLU transform: bottom -> bottom
//   data size: total number of nodes (N * C * H * W or something)
//   if option->negative_slope = 0, perform ReLU
//                             > 0, perform soft ReLU
void relu_forward_inplace(Tensor* const bottom,
                          const ReluOption* const option);

void relu_shape(const Tensor* const bottom,
                Tensor* const top);



// --------------------------------------------------------------------------
// top-n proposal generation
//   struct ProposalOption
//   proposal_forward
//   proposal_shape
//   generate_anchors
// --------------------------------------------------------------------------

typedef struct ProposalOption_
{
  real* scales;
  real* ratios;
  int num_scales;
  int num_ratios;
  int num_concats;
  int base_size;
  int feat_stride;
  int min_size;
  int pre_nms_topn;
  int post_nms_topn;
  real nms_thresh;
} ProposalOption;

// proposal: bottom -> top
//   bottom: 2 x num_anchors x H x W tensor
//     bottom[0, n, h, w] = foreground score of anchor n at node (h, w)
//     bottom[1, n, h, w] = background score of anchor n at node (h, w)
//   d_anchor: num_anchors x 4 x H x W tensor
//     d_anchor[n, :, h, w] = gradient (dx, dy, d(log w), d(log h))
//                            of anchor n at center location (h, w)
//   img_info: 4 x 1 tensor,  (img_H, img_W, min_box_W, min_box_H)
//     img_H, img_W: raw image height & width
//     min_box_W: minimum box width in raw image
//     min_box_H: minimum box height in raw image
//   top: num_RoIs x 4 tensor,  (x1, y1, x2, y2) of each RoI
//   anchors: num_anchors * 4 array,  (x1, y1, x2, y2) of each anchor
//   4 temporary arrays
//     proposals: all box proposals with their scores
//       "num_boxes x 5" array,  (x1, y1, x2, y2, score) for each box
//       TODO: always stored in main memory due to implementation issue
//     keep: indices of proposals to be retrieved as RoIs
//       "num_rois x 1" array,  keep[i]: index of i-th RoI in proposals
//       TODO: always stored in main memory due to implementation issue
//     proposals_dev: GPU memory space, required in GPU mode
//     keep_dev: GPU memory space, required in GPU mode
void proposal_forward(const Tensor* const bottom4d,
                      const Tensor* const d_anchor4d,
                      const Tensor* const img_info1d,
                      Tensor* const top2d,
                      const real* const anchors,
                      real* const proposals,
                      int* const keep,
                      real* const proposals_dev,
                      int* const keep_dev,
                      const ProposalOption* const option);

void proposal_shape(const Tensor* const bottom4d,
                    Tensor* const top2d,
                    int* const proposals_size,
                    int* const keep_size,
                    const ProposalOption* const option);

// given a base box, enumerate transformed boxes of varying sizes and ratios
//   option->base_size: base box's width & height (i.e., base box is square)
//   option->scales: "option->num_scales x 1" array
//                   varying scale factor for base box
//   option->ratios: "option->num_ratios x 1" array
//                   varying height-width ratio
//   option->num_concats: repeat count of anchor set generation
//                        (required for separated RPN)
//   anchors: "num_boxes x 4" array,  (x1, y1, x2, y2) for each box
//     num_boxes = total number of transformations
//         = option->num_scales * option->num_ratios * option->num_concats
void generate_anchors(real* const anchors,
                      const ProposalOption* const option);



// --------------------------------------------------------------------------
// dropout
//   struct DropoutOption
//   dropout_forward
//   dropout_forward_inplace
//   dropout_shape
// --------------------------------------------------------------------------

typedef struct DropoutOption_
{
  int scaled;
  int test;
  real threshold;
} DropoutOption;

// dropout transform: bottom -> top
//   if option->scaled = 1, perform scaled dropout
//   if option->test = 1, perform testing-time dropout
//   if both = 1, perform testing-time scaled dropout,
//                which is actually do nothing:  top[i] = bottom[i]
//   if both = 0, perform dropout
//   data size: total number of nodes (N * C * H * W or something)
//   mask: data_size x 1 temporary array
void dropout_forward(const Tensor* const bottom,
                     unsigned int* const mask,
                     Tensor* const top,
                     const DropoutOption* const option);

// in-place dropout transform: bottom -> bottom
void dropout_forward_inplace(Tensor* const bottom,
                             unsigned int* const mask,
                             const DropoutOption* const option);

void dropout_shape(const Tensor* const bottom,
                   Tensor* const top);



// --------------------------------------------------------------------------
// concat
//   concat_forward
//   concat_shape
// --------------------------------------------------------------------------

// concat: bottom[0], bottom[1], ..., bottom[M-1] -> top
//   M = num_bottoms
//   bottom[m]: C_m x H x W  (C_m may different from each other)
//   top: sum(C_m) x H x W  (channel-wise concatenation)
void concat_forward(const Tensor* const bottom3d[],
                    Tensor* const top3d,
                    const int num_bottoms);

void concat_shape(const Tensor* const bottom3d[],
                  Tensor* const top3d,
                  const int num_bottoms);



// --------------------------------------------------------------------------
// softmax
//   softmax_forward
//   softmax_inplace_forward
//   softmax_shape
// --------------------------------------------------------------------------

// channel-wise softmax transform: bottom3d (N x C x D) -> top3d (N x C x D)
//   top[n][c][d] = exp(bottom[n][c][d]) / sum_c(exp(bottom[n][c][d]))
//   temp_data: N * D array,  temporary space for channel-wise sum or max
//     e.g., temp_data[n][d] = sum_c(exp(bottom[n][c][d]))
void softmax_forward(const Tensor* const bottom3d,
                     Tensor* const top3d,
                     real* const temp_data);

// channel-wise in-place softmax transform:
//   bottom[n][c][d] = exp(bottom[n][c][d]) / sum_c(exp(bottom[n][c][d]))
void softmax_inplace_forward(Tensor* const bottom3d,
                             real* const temp_data);

void softmax_shape(const Tensor* const bottom3d,
                   Tensor* const top3d,
                   int* const temp_size);



// --------------------------------------------------------------------------
// NMS operation
//   nms
// --------------------------------------------------------------------------

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
         const real nms_thresh, const int max_num_out);


#endif // endifndef PVA_DL_LAYER_H
