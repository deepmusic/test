/*!
 * Copyright (c) 2016 by Kye-Hyeon Kim
 * \file proposal.cu
 * \brief proposal operator
*/
#include "./proposal-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(ProposalParam param) {
  return new ProposalOp<gpu>(param);
}
}  // namespace op
}  // namespace mxnet
