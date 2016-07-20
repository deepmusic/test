// --------------------------------------------------------------------------
// layer creators for building deep networks
//   callable from external environments via shared library
// --------------------------------------------------------------------------

#ifndef PVA_DL_LAYER_FACTORY_H
#define PVA_DL_LAYER_FACTORY_H

#include "core/net.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer* add_data_layer(Net* const net,
                      const char* const layer_name,
                      const char* const data_name,
                      const char* const img_info_name);

Layer* add_conv_layer(Net* const net,
                      const char* const layer_name,
                      const char* const bottom_name,
                      const char* const top_name,
                      const char* const weight_name,
                      const char* const bias_name,
                      const int num_group, const int num_output,
                      const int kernel_h, const int kernel_w,
                      const int stride_h, const int stride_w,
                      const int pad_h, const int pad_w,
                      const int bias_term);

Layer* add_deconv_layer(Net* const net,
                        const char* const layer_name,
                        const char* const bottom_name,
                        const char* const top_name,
                        const char* const weight_name,
                        const char* const bias_name,
                        const int num_group, const int num_output,
                        const int kernel_h, const int kernel_w,
                        const int stride_h, const int stride_w,
                        const int pad_h, const int pad_w,
                        const int bias_term);

Layer* add_fc_layer(Net* const net,
                    const char* const layer_name,
                    const char* const bottom_name,
                    const char* const top_name,
                    const char* const weight_name,
                    const char* const bias_name,
                    const int num_output,
                    const int bias_term);

Layer* add_max_pool_layer(Net* const net,
                          const char* const layer_name,
                          const char* const bottom_name,
                          const char* const top_name,
                          const int kernel_h, const int kernel_w,
                          const int stride_h, const int stride_w,
                          const int pad_h, const int pad_w);

Layer* add_scale_const_layer(Net* const net,
                             const char* const layer_name,
                             const char* const bottom_name,
                             const char* const top_name,
                             const real weight,
                             const real bias,
                             const int bias_term);

Layer* add_scale_channel_layer(Net* const net,
                               const char* const layer_name,
                               const char* const bottom_name,
                               const char* const top_name,
                               const char* const weight_name,
                               const char* const bias_name,
                               const int bias_term);

Layer* add_concat_layer(Net* const net,
                        const char* const layer_name,
                        const char* const p_bottom_names[],
                        const char* const top_name,
                        const int num_bottoms);

Layer* add_eltwise_layer(Net* const net,
                         const char* const layer_name,
                         const char* const p_bottom_names[],
                         const char* const top_name,
                         const int num_bottoms);

Layer* add_relu_layer(Net* const net,
                      const char* const layer_name,
                      const char* const bottom_name,
                      const char* const top_name,
                      const real negative_slope);

Layer* add_dropout_layer(Net* const net,
                         const char* const layer_name,
                         const char* const bottom_name,
                         const char* const top_name,
                         const real dropout_ratio,
                         const int is_test_phase,
                         const int is_scaled_dropout);

Layer* add_reshape_layer(Net* const net,
                         const char* const layer_name,
                         const char* const bottom_name,
                         const char* const top_name,
                         const int shape[],
                         const int ndim);

Layer* add_softmax_layer(Net* const net,
                         const char* const layer_name,
                         const char* const bottom_name,
                         const char* const top_name,
                         const int channel_axis);

Layer* add_proposal_layer(Net* const net,
                          const char* const layer_name,
                          const char* const score_name,
                          const char* const bbox_name,
                          const char* const img_info_name,
                          const char* const top_name,
                          real anchor_scales[], const int num_anchor_scales,
                          real anchor_ratios[], const int num_anchor_ratios,
                          const int feat_stride,
                          const int base_size, const int min_size,
                          const int pre_nms_topn, const int post_nms_topn,
                          const real nms_thresh,
                          const int bbox_vote, const real vote_thresh);

Layer* add_roipool_layer(Net* const net,
                         const char* const layer_name,
                         const char* const rcnn_input_name,
                         const char* const roi_name,
                         const char* const top_name,
                         const int pooled_h, const int pooled_w,
                         const real spatial_scale,
                         const int flatten_shape);

Layer* add_odout_layer(Net* const net,
                       const char* const layer_name,
                       const char* const score_name,
                       const char* const bbox_name,
                       const char* const roi_name,
                       const char* const img_info_name,
                       const char* const top_name,
                       const int min_size, const int pre_nms_topn,
                       const real score_thresh, const real nms_thresh,
                       const int bbox_vote, const real vote_thresh);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end PVA_DL_LAYER_FACTORY_H
