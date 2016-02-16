/*!
 * Copyright (c) 2016 by Kye-Hyeon Kim
 * \file proposal.cc
 * \brief proposal operator
*/
#include "./proposal-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(ProposalParam param) {
  return new ProposalOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from static_operator_common.h
Operator* ProposalProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(ProposalParam);

MXNET_REGISTER_OP_PROPERTY(Proposal, ProposalProp)
.describe("Generate proposals of region-of-interest (RoI) in an input plane.")
.add_argument("data", "Symbol[]", "[class score tensor, predicted bounding box]")
.add_argument("im_info", "Symbol", "Raw image size (h, w, h_scale, w_scale)")
.add_arguments(ProposalParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
