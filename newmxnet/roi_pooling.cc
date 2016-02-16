/*!
 * Copyright (c) 2016 by Kye-Hyeon Kim
 * \file roi_pooling.cc
 * \brief roi pooling operator
*/
#include "./roi_pooling-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(ROIPoolingParam param) {
  return new ROIPoolingOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from static_operator_common.h
Operator* ROIPoolingProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(ROIPoolingParam);

MXNET_REGISTER_OP_PROPERTY(ROIPooling, ROIPoolingProp)
.describe("Resize regions of interest in an input plane to a fixed size by MAX pooling.")
.add_argument("data", "Symbol[]", "[input tensor, regions of interest]")
.add_arguments(ROIPoolingParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
