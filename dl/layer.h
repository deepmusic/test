#ifndef PVA_DL_LAYERS_H
#define PVA_DL_LAYERS_H

#include <cblas.h>
#include <string.h>
#include <stdio.h>

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

typedef struct ConvOption_
{
  int kernel_h, kernel_w;
  int pad_h, pad_w;
  int stride_h, stride_w;
  int num_groups;
  int bias;
  void* handle;
} ConvOption;

int flatten_size(const Tensor* tensor);

inline int flatten_size(const Tensor* tensor)
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

#endif // PVA_DL_LAYERS_H
