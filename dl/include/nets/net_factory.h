// --------------------------------------------------------------------------
// network creators
// --------------------------------------------------------------------------

#ifndef PVA_DL_NET_FACTORY_H
#define PVA_DL_NET_FACTORY_H

#include "layers/layer_factory.h"

void setup_shared_cnn(Net* const net,
                      const int input_size);

void setup_shared_cnn_light(Net* const net,
                            const int input_size);

void setup_faster_rcnn(Net* const net,
                       const char* const rpn_input_name,
                       const char* const rcnn_input_name,
                       const int rpn_channels,
                       const int rpn_kernel_h, const int rpn_kernel_w,
                       const int fc_compress,
                       const int fc6_channels, const int fc7_channels,
                       const int pre_nms_topn, const int post_nms_topn);

#endif // end PVA_DL_NET_FACTORY_H
