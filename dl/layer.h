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
// --------------------------------------------------------------------------

typedef float real;
#define g_max_num_items 128
#define g_max_ndim 5

typedef struct Tensor_
{
  real* data;
  int num_items;
  int ndim;
  int shape[g_max_num_items][g_max_ndim];
} Tensor;

// total number of elements in a tensor
int flatten_size(const Tensor* const tensor);

inline int flatten_size(const Tensor* const tensor)
{
  int total_size = 0;
  for (int n = 0; n < tensor->num_items; ++n) {
    int size = 1;
    for (int d = 0; d < tensor->ndim; ++d) {
      size *= tensor->shape[n][d];
    }
    total_size += size;
  }
  return total_size;
}



// --------------------------------------------------------------------------
// convolution & deconvolution
//   struct ConvOption
//   conv_forward
//   deconv_forward
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



// --------------------------------------------------------------------------
// fully-connected
//   struct FCOption
//   fc_forward
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



// --------------------------------------------------------------------------
// pooling
//   struct PoolOption
//   pool_forward
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



// --------------------------------------------------------------------------
// RoI pooling
//   struct ROIPoolOption
//   roipool_forward
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



// --------------------------------------------------------------------------
// ReLU transform
//   struct ReluOption
//   relu_forward
//   relu_forward_inplace
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



// --------------------------------------------------------------------------
// top-n proposal generation
//   struct ProposalOption
//   proposal_forward
// --------------------------------------------------------------------------

typedef struct ProposalOption_
{
  int num_concats;
  real* ratios;
  int num_ratios;
  real* scales;
  int num_scales;
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
void proposal_forward(const Tensor* const bottom4d,
                      const Tensor* const d_anchor4d,
                      const Tensor* const img_info1d,
                      Tensor* const top2d,
                      const real* const anchors,
                      const ProposalOption* const option);


#endif // endifndef PVA_DL_LAYER_H
