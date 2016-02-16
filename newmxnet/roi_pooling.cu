/*!
 * Copyright (c) 2016 by Kye-Hyeon Kim
 * \file roi_pooling.cu
 * \brief roi pooling operator
*/
#include "./roi_pooling-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(ROIPoolingParam param) {
  return new ROIPoolingOp<gpu>(param);
}
}  // namespace op
}  // namespace mxnet
