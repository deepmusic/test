#ifndef PVA_DL_LAYER_H
#define PVA_DL_LAYER_H

// simple math operations
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


// tensor data structure & some functions
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


// convolution & deconvolution
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
//   temp: G * C * kernel_h * kernel_w * H' * W'
//   const: H' * W',  const[i] = 1 for all i
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
//   temp: G * C * kernel_h * kernel_w * H' * W'
//   const: H * W,  const[i] = 1 for all i
void deconv_forward(const Tensor* const bottom3d,
                    Tensor* const top3d,
                    const Tensor* const weight5d,
                    const Tensor* const bias1d,
                    real* const temp_data,
                    const real* const const_data,
                    const ConvOption* const option);


// pooling
typedef struct PoolOption_
{
  int kernel_h, kernel_w;
  int pad_h, pad_w;
  int stride_h, stride_w;
} PoolOption;


// RoI pooling
typedef struct ROIPoolOption_
{
  int pooled_height;
  int pooled_width;
  real spatial_scale;
} ROIPoolOption;


// ReLU transform
typedef struct ReluOption_
{
  real negative_slope;
} ReluOption;


// top-n proposal generation
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
//   pred_box: num_anchors x 4 x H x W tensor
//     pred_box[n, :, h, w] = predicted box (d x1, d y1, d log w, d log h)
//                            of anchor n at pixel (h, w)
//   img_info: 4 x 1 tensor,  (w, h, min_w, min_h) of raw image
//     min_w: minimum box width in raw image
//     min_h: minimum box height in raw image
//   top: num_RoIs x 4 tensor,  (x1, y1, x2, y2) of each RoI
//   anchors: num_anchors * 4 array,  (x1, y1, x2, y2) of each anchor
void proposal_forward(const Tensor* const bottom4d,
                      const Tensor* const pred_box4d,
                      const Tensor* const img_info1d,
                      Tensor* const top2d,
                      const real* const anchors,
                      const ProposalOption* const option);

#endif // endifndef PVA_DL_LAYER_H
