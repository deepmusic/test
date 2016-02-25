#ifndef PVA_DL_LAYERS_H
#define PVA_DL_LAYERS_H

typedef float real;

#define ABS(x)  ((x) > 0 ? (x) : (-(x)))
#define MIN(x, y)  ((x) < (y) ? (x) : (y))
#define MAX(x, y)  ((x) > (y) ? (x) : (y))
#define DIV_THEN_CEIL(x, y)  (((x) + (y) - 1) / (y))
#define ROUND(x)  ((int)((x) + 0.5f))

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
  int size = 0;
  for (int n = 0; n < tensor->num_items; ++n) {
    int size_n = 1;
    for (int d = 0; d < tensor->ndim; ++d)
      size_n *= tensor->shape[n][d];
    size += size_n;
  }
  return size;
}

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

typedef struct PoolOption_
{
  int kernel_h, kernel_w;
  int pad_h, pad_w;
  int stride_h, stride_w;
} PoolOption;

typedef struct ROIPoolOption_
{
  int pooled_height;
  int pooled_width;
  real spatial_scale;
} ROIPoolOption;

typedef struct ReluOption_
{
  real negative_slope;
} ReluOption;

#endif // PVA_DL_LAYERS_H
