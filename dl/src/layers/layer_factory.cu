#include "layers/layer_factory.h"
#include "layers/operator.h"
#include <string.h>

Layer* add_data_layer(Net* const net,
                      const char* const layer_name,
                      const char* const data_name,
                      const char* const img_info_name)
{
  Layer* const layer = add_layer(net, layer_name);
  Tensor* const data = add_tensor(net, data_name);
  Tensor* const img_info = add_tensor(net, img_info_name);

  add_top(layer, data);
  add_top(layer, img_info);

  img_info->data_type = PRIVATE_DATA;
  init_input_layer(net, data, img_info);

  return layer;
}

Layer* add_chain_layer(Net* const net,
                       const char* const layer_name,
                       const char* const bottom_name,
                       const char* const top_name)
{
  Layer* const layer = add_layer(net, layer_name);
  Tensor* const bottom = get_tensor_by_name(net, bottom_name);
  Tensor* const top = (strcmp(bottom_name, top_name) != 0)
                      ? add_tensor(net, top_name) : bottom;

  add_bottom(layer, bottom);
  add_top(layer, top);

  return layer;
}

Layer* add_hub_layer(Net* const net,
                     const char* const layer_name,
                     const char* const p_bottom_names[],
                     const char* const top_name,
                     const int num_bottoms)
{
  Layer* const layer = add_layer(net, layer_name);

  for (int i = 0; i < num_bottoms; ++i) {
    add_bottom(layer, get_tensor_by_name(net, p_bottom_names[i]));
  }
  add_top(layer, add_tensor(net, top_name));

  return layer;
}

Layer* add_param_layer(Net* const net,
                       const char* const layer_name,
                       const char* const bottom_name,
                       const char* const top_name,
                       const char* const weight_name,
                       const char* const bias_name,
                       const int bias_term)
{
  Layer* const layer =
      add_chain_layer(net, layer_name, bottom_name, top_name);

  if (!weight_name) {
    char temp_name[MAX_NAME_LEN];
    sprintf(temp_name, "%s_param%d", layer_name, layer->num_params);
    add_param(layer, add_tensor(net, temp_name));
  }
  else {
    add_param(layer, find_or_add_tensor(net, weight_name));
  }

  if (bias_term) {
    if (!bias_name) {
      char temp_name[MAX_NAME_LEN];
      sprintf(temp_name, "%s_param%d", layer_name, layer->num_params);
      add_param(layer, add_tensor(net, temp_name));
    }
    else {
      add_param(layer, find_or_add_tensor(net, bias_name));
    }
  }

  return layer;
}

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
                      const int bias_term)
{
  Layer* const layer = add_param_layer(net, layer_name,
      bottom_name, top_name, weight_name, bias_name, bias_term);

  layer->option.kernel_h = kernel_h;
  layer->option.kernel_w = kernel_w;
  layer->option.stride_h = stride_h;
  layer->option.stride_w = stride_w;
  layer->option.pad_h = pad_h;
  layer->option.pad_w = pad_w;
  layer->option.num_output = num_output;
  layer->option.group = num_group;
  layer->option.bias = bias_term;
  layer->option.handle = (void*)&net->blas_handle;

  layer->f_forward = forward_conv_layer;
  layer->f_shape = shape_conv_layer;

  return layer;
}

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
                        const int bias_term)
{
  Layer* const layer = add_conv_layer(net, layer_name,
      bottom_name, top_name, weight_name, bias_name, num_group, num_output,
      kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, bias_term);

  layer->f_forward = forward_deconv_layer;
  layer->f_shape = shape_deconv_layer;
  layer->f_free = free_deconv_layer;

  malloc_deconv_layer(net, layer);

  return layer;
}

Layer* add_fc_layer(Net* const net,
                    const char* const layer_name,
                    const char* const bottom_name,
                    const char* const top_name,
                    const char* const weight_name,
                    const char* const bias_name,
                    const int num_output,
                    const int bias_term)
{
  Layer* const layer = add_param_layer(net, layer_name,
      bottom_name, top_name, weight_name, bias_name, bias_term);

  layer->option.num_output = num_output;
  layer->option.bias = bias_term;
  layer->option.handle = (void*)&net->blas_handle;

  layer->f_forward = forward_fc_layer;
  layer->f_shape = shape_fc_layer;

  return layer;
}

Layer* add_max_pool_layer(Net* const net,
                          const char* const layer_name,
                          const char* const bottom_name,
                          const char* const top_name,
                          const int kernel_h, const int kernel_w,
                          const int stride_h, const int stride_w,
                          const int pad_h, const int pad_w)
{
  Layer* const layer =
      add_chain_layer(net, layer_name, bottom_name, top_name);

  layer->option.kernel_h = kernel_h;
  layer->option.kernel_w = kernel_w;
  layer->option.stride_h = stride_h;
  layer->option.stride_w = stride_w;
  layer->option.pad_h = pad_h;
  layer->option.pad_w = pad_w;

  layer->f_forward = forward_max_pool_layer;
  layer->f_shape = shape_pool_layer;

  return layer;
}

Layer* add_scale_const_layer(Net* const net,
                             const char* const layer_name,
                             const char* const bottom_name,
                             const char* const top_name,
                             const real weight,
                             const real bias,
                             const int bias_term)
{
  Layer* const layer =
      add_chain_layer(net, layer_name, bottom_name, top_name);

  layer->option.scale_weight = weight;
  layer->option.scale_bias = bias;
  layer->option.bias = bias_term;

  layer->f_forward = forward_scale_const_layer;
  layer->f_shape = shape_scale_const_layer;

  return layer;
}

Layer* add_scale_channel_layer(Net* const net,
                               const char* const layer_name,
                               const char* const bottom_name,
                               const char* const top_name,
                               const char* const weight_name,
                               const char* const bias_name,
                               const int bias_term)
{
  Layer* const layer = add_param_layer(net, layer_name,
      bottom_name, top_name, weight_name, bias_name, bias_term);

  layer->option.bias = bias_term;

  layer->f_forward = forward_scale_channel_layer;
  layer->f_shape = shape_scale_channel_layer;

  return layer;
}

Layer* add_concat_layer(Net* const net,
                        const char* const layer_name,
                        const char* const p_bottom_names[],
                        const char* const top_name,
                        const int num_bottoms)
{
  Layer* const layer = add_hub_layer(net, layer_name,
      p_bottom_names, top_name, num_bottoms);

  layer->f_forward = forward_concat_layer;
  layer->f_shape = shape_concat_layer;

  return layer;
}

Layer* add_eltwise_layer(Net* const net,
                         const char* const layer_name,
                         const char* const p_bottom_names[],
                         const char* const top_name,
                         const int num_bottoms)
{
  Layer* const layer = add_hub_layer(net, layer_name,
      p_bottom_names, top_name, num_bottoms);

  layer->f_forward = forward_eltwise_sum_layer;
  layer->f_shape = shape_eltwise_layer;

  return layer;
}

Layer* add_relu_layer(Net* const net,
                      const char* const layer_name,
                      const char* const bottom_name,
                      const char* const top_name,
                      const real negative_slope)
{
  Layer* const layer =
      add_chain_layer(net, layer_name, bottom_name, top_name);

  layer->option.negative_slope = negative_slope;

  layer->f_forward = forward_relu_layer;
  layer->f_shape = shape_relu_layer;

  return layer;
}

Layer* add_dropout_layer(Net* const net,
                         const char* const layer_name,
                         const char* const bottom_name,
                         const char* const top_name,
                         const real dropout_ratio,
                         const int is_test_phase,
                         const int is_scaled_dropout)
{
  Layer* const layer =
      add_chain_layer(net, layer_name, bottom_name, top_name);

  layer->option.dropout_ratio = dropout_ratio;
  layer->option.test_dropout = is_test_phase;
  layer->option.scaled_dropout = is_scaled_dropout;

  layer->f_forward = forward_dropout_layer;
  layer->f_shape = shape_dropout_layer;

  return layer;
}

Layer* add_reshape_layer(Net* const net,
                         const char* const layer_name,
                         const char* const bottom_name,
                         const char* const top_name,
                         const int shape[],
                         const int ndim)
{
  Layer* const layer =
      add_chain_layer(net, layer_name, bottom_name, top_name);

  for (int i = 0; i < ndim; ++i) {
    layer->option.reshape[i] = shape[i];
  }
  layer->option.reshape_ndim = ndim;

  layer->f_forward = forward_reshape_layer;
  layer->f_shape = shape_reshape_layer;

  return layer;
}

Layer* add_softmax_layer(Net* const net,
                         const char* const layer_name,
                         const char* const bottom_name,
                         const char* const top_name,
                         const int channel_axis)
{
  Layer* const layer =
      add_chain_layer(net, layer_name, bottom_name, top_name);

  layer->option.channel_axis = channel_axis;

  layer->f_forward = forward_softmax_layer;
  layer->f_shape = shape_softmax_layer;

  return layer;
}

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
                          const int bbox_vote, const real vote_thresh)
{
  Layer* const layer = add_layer(net, layer_name);

  add_bottom(layer, get_tensor_by_name(net, score_name));
  add_bottom(layer, get_tensor_by_name(net, bbox_name));
  add_bottom(layer, get_tensor_by_name(net, img_info_name));
  add_top(layer, add_tensor(net, top_name));

  layer->option.num_anchor_scales = num_anchor_scales;
  layer->option.num_anchor_ratios = num_anchor_ratios;
  layer->option.base_size = base_size;
  layer->option.feat_stride = feat_stride;
  layer->option.min_size = min_size;
  layer->option.pre_nms_topn = pre_nms_topn;
  layer->option.post_nms_topn = post_nms_topn;
  layer->option.nms_thresh = nms_thresh;
  layer->option.bbox_vote = bbox_vote;
  layer->option.vote_thresh = vote_thresh;
  layer->option.anchor_scales = anchor_scales;
  layer->option.anchor_ratios = anchor_ratios;

  layer->f_forward = forward_proposal_layer;
  layer->f_shape = shape_proposal_layer;
  layer->f_free = free_proposal_layer;

  malloc_proposal_layer(net, layer);

  return layer;
}

Layer* add_roipool_layer(Net* const net,
                         const char* const layer_name,
                         const char* const rcnn_input_name,
                         const char* const roi_name,
                         const char* const top_name,
                         const int pooled_h, const int pooled_w,
                         const real spatial_scale,
                         const int flatten_shape)
{
  Layer* const layer = add_layer(net, layer_name);

  add_bottom(layer, get_tensor_by_name(net, rcnn_input_name));
  add_bottom(layer, get_tensor_by_name(net, roi_name));
  add_top(layer, add_tensor(net, top_name));

  layer->option.pooled_height = pooled_h;
  layer->option.pooled_width = pooled_w;
  layer->option.spatial_scale = spatial_scale;
  layer->option.flatten_shape = flatten_shape;

  layer->f_forward = forward_roipool_layer;
  layer->f_shape = shape_roipool_layer;

  return layer;
}

Layer* add_odout_layer(Net* const net,
                       const char* const layer_name,
                       const char* const score_name,
                       const char* const bbox_name,
                       const char* const roi_name,
                       const char* const img_info_name,
                       const char* const top_name,
                       const int min_size, const int pre_nms_topn,
                       const real score_thresh, const real nms_thresh,
                       const int bbox_vote, const real vote_thresh)
{
  Layer* const layer = add_layer(net, layer_name);

  add_bottom(layer, get_tensor_by_name(net, score_name));
  add_bottom(layer, get_tensor_by_name(net, bbox_name));
  add_bottom(layer, get_tensor_by_name(net, roi_name));
  add_bottom(layer, get_tensor_by_name(net, img_info_name));
  add_top(layer, add_tensor(net, top_name));

  layer->option.min_size = min_size;
  layer->option.pre_nms_topn = pre_nms_topn;
  layer->option.score_thresh = score_thresh;
  layer->option.nms_thresh = nms_thresh;
  layer->option.bbox_vote = bbox_vote;
  layer->option.vote_thresh = vote_thresh;

  layer->f_forward = forward_odout_layer;
  layer->f_shape = shape_odout_layer;
  layer->f_free = free_odout_layer;

  malloc_odout_layer(net, layer);

  return layer;
}
