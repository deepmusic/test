#include "layer.h"
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

  img_info->data_type = CPU_DATA;
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

  layer->option.num_bottoms = num_bottoms;

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
                      const int bias_term,
                      const int do_relu)
{
  Layer* const layer = add_param_layer(net, layer_name,
      bottom_name, top_name, weight_name, bias_name, bias_term);

  layer->option.kernel_h = kernel_h;
  layer->option.kernel_w = kernel_w;
  layer->option.stride_h = stride_h;
  layer->option.stride_w = stride_w;
  layer->option.pad_h = pad_h;
  layer->option.pad_w = pad_w;
  layer->option.out_channels = num_output;
  layer->option.num_groups = num_group;
  layer->option.bias = bias_term;
  layer->option.handle = (void*)&net->blas_handle;

  layer->f_forward[0] = forward_conv_layer;
  if (do_relu) {
    layer->f_forward[1] = forward_inplace_relu_layer;
  }
  layer->f_shape[0] = shape_conv_layer;

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
                        const int bias_term,
                        const int do_relu)
{
  Layer* const layer = add_conv_layer(net, layer_name,
      bottom_name, top_name, weight_name, bias_name, num_group, num_output,
      kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
      bias_term, do_relu);

  layer->f_forward[0] = forward_deconv_layer;
  layer->f_shape[0] = shape_deconv_layer;

  return layer;
}

Layer* add_fc_layer(Net* const net,
                    const char* const layer_name,
                    const char* const bottom_name,
                    const char* const top_name,
                    const char* const weight_name,
                    const char* const bias_name,
                    const int num_output,
                    const int bias_term,
                    const int do_relu)
{
  Layer* const layer = add_param_layer(net, layer_name,
      bottom_name, top_name, weight_name, bias_name, bias_term);

  layer->option.out_channels = num_output;
  layer->option.bias = bias_term;
  layer->option.handle = (void*)&net->blas_handle;

  layer->f_forward[0] = forward_fc_layer;
  if (do_relu) {
    layer->f_forward[1] = forward_inplace_relu_layer;
  }
  layer->f_shape[0] = shape_fc_layer;

  return layer;
}

Layer* add_pool_layer(Net* const net,
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

  layer->f_forward[0] = forward_pool_layer;
  layer->f_shape[0] = shape_pool_layer;

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

  layer->f_forward[0] = forward_scale_const_layer;
  layer->f_shape[0] = shape_scale_const_layer;

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

  layer->f_forward[0] = forward_scale_channel_layer;
  layer->f_shape[0] = shape_scale_channel_layer;

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

  layer->f_forward[0] = forward_concat_layer;
  layer->f_shape[0] = shape_concat_layer;

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

  layer->f_forward[0] = forward_eltwise_sum_layer;
  layer->f_shape[0] = shape_eltwise_layer;

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

  layer->f_forward[0] = forward_relu_layer;
  layer->f_shape[0] = shape_relu_layer;

  return layer;
}

Layer* add_dropout_layer(Net* const net,
                         const char* const layer_name,
                         const char* const bottom_name,
                         const char* const top_name,
                         const int is_test,
                         const int is_scaled)
{
  Layer* const layer =
      add_chain_layer(net, layer_name, bottom_name, top_name);

  layer->option.test = is_test;
  layer->option.scaled = is_scaled;

  layer->f_forward[0] = forward_dropout_layer;
  layer->f_shape[0] = shape_dropout_layer;

  return layer;
}

static
void setup_inception_sub(Net* const net,
                         const char* const sub_name,
                         const char* const input_name,
                         const int out_channels[],
                         const int stride)
{
  char bottom_name[MAX_NAME_LEN];
  char top_name[MAX_NAME_LEN];
  char top_name2[MAX_NAME_LEN];

  if (stride > 1) {
    // pool1
    sprintf(top_name, "%s_pool1", sub_name);
    add_pool_layer(net, top_name, input_name, top_name,
                   3, 3, stride, stride, 0, 0);
    // conv1
    sprintf(bottom_name, "%s_pool1", sub_name);
    sprintf(top_name, "%s_conv1", sub_name);
    add_conv_layer(net, top_name, bottom_name, top_name, NULL, NULL,
                     1, out_channels[0], 1, 1, 1, 1, 0, 0, 1, 0);
  }
  else {
    // conv1
    sprintf(top_name, "%s_conv1", sub_name);
    add_conv_layer(net, top_name, input_name, top_name, NULL, NULL,
                     1, out_channels[0], 1, 1, 1, 1, 0, 0, 1, 0);
  }
  sprintf(top_name2, "%s_relu1", sub_name);
  add_relu_layer(net, top_name2, top_name, top_name, 0);

  // conv3_1
  sprintf(top_name, "%s_conv3_1", sub_name);
  add_conv_layer(net, top_name, input_name, top_name, NULL, NULL,
                   1, out_channels[1], 1, 1, 1, 1, 0, 0, 1, 0);
  sprintf(top_name2, "%s_relu3_1", sub_name);
  add_relu_layer(net, top_name2, top_name, top_name, 0);

  // conv3_2
  sprintf(bottom_name, "%s_conv3_1", sub_name);
  sprintf(top_name, "%s_conv3_2", sub_name);
  add_conv_layer(net, top_name, bottom_name, top_name, NULL, NULL,
                   1, out_channels[2], 3, 3, stride, stride, 1, 1, 1, 0);
  sprintf(top_name2, "%s_relu3_2", sub_name);
  add_relu_layer(net, top_name2, top_name, top_name, 0);

  // conv5_1
  sprintf(top_name, "%s_conv5_1", sub_name);
  add_conv_layer(net, top_name, input_name, top_name, NULL, NULL,
                   1, out_channels[3], 1, 1, 1, 1, 0, 0, 1, 0);
  sprintf(top_name2, "%s_relu5_1", sub_name);
  add_relu_layer(net, top_name2, top_name, top_name, 0);

  // conv5_2
  sprintf(bottom_name, "%s_conv5_1", sub_name);
  sprintf(top_name, "%s_conv5_2", sub_name);
  add_conv_layer(net, top_name, bottom_name, top_name, NULL, NULL,
                   1, out_channels[4], 3, 3, 1, 1, 1, 1, 1, 0);
  sprintf(top_name2, "%s_relu5_2", sub_name);
  add_relu_layer(net, top_name2, top_name, top_name, 0);

  // conv5_2
  sprintf(bottom_name, "%s_conv5_2", sub_name);
  sprintf(top_name, "%s_conv5_3", sub_name);
  add_conv_layer(net, top_name, bottom_name, top_name, NULL, NULL,
                   1, out_channels[5], 3, 3, stride, stride, 1, 1, 1, 0);
  sprintf(top_name2, "%s_relu5_3", sub_name);
  add_relu_layer(net, top_name2, top_name, top_name, 0);

  // concat
  {
    char bottom_names[3][MAX_NAME_LEN];
    char* const p_bottom_names[3] = {
        bottom_names[0], bottom_names[1], bottom_names[2]
    };
    sprintf(p_bottom_names[0], "%s_conv1", sub_name);
    sprintf(p_bottom_names[1], "%s_conv3_2", sub_name);
    sprintf(p_bottom_names[2], "%s_conv5_3", sub_name);
    sprintf(top_name, "%s", sub_name);
    add_concat_layer(net, top_name, p_bottom_names, top_name, 3);
  }
}

void setup_hyper_sub(Net* const net,
                     const char* const downsample_name,
                     const char* const as_is_name,
                     const char* const upsample_name,
                     const int out_channels[])
{
  const char* const p_hyper_names[3] = {
      "downsample", as_is_name, "upsample"
  };

  // downsample
  add_pool_layer(net, p_hyper_names[0], downsample_name, p_hyper_names[0],
                 3, 3, 2, 2, 0, 0);

  // upsample
  add_deconv_layer(net, p_hyper_names[2], upsample_name, p_hyper_names[2],
                   NULL, NULL, out_channels[0], out_channels[0],
                   4, 4, 2, 2, 1, 1, 0, 0);

  // concat
  add_concat_layer(net, "concat", p_hyper_names, "concat", 3);

  // convf
  add_conv_layer(net, "convf", "concat", "convf", NULL, NULL,
                 1, out_channels[1], 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "reluf", "convf", "convf", 0);
}

void setup_inception(Net* const net)
{
  // data
  add_data_layer(net, "data", "data", "img_info");

  // conv1
  add_conv_layer(net, "conv1", "data", "conv1", NULL, NULL,
                 1, 32, 4, 4, 2, 2, 1, 1, 1, 0);
  add_relu_layer(net, "relu1", "conv1", "conv1", 0);
  get_tensor_by_name(net, "conv1")->data_type = PRIVATE_DATA;

  // conv2
  add_conv_layer(net, "conv2", "conv1", "conv2", NULL, NULL,
                 1, 48, 3, 3, 2, 2, 1, 1, 1, 0);
  add_relu_layer(net, "relu2", "conv2", "conv2", 0);

  // conv3
  add_conv_layer(net, "conv3", "conv2", "conv3", NULL, NULL,
                 1, 96, 3, 3, 2, 2, 1, 1, 1, 0);
  add_relu_layer(net, "relu3", "conv3", "conv3", 0);
  get_tensor_by_name(net, "conv3")->data_type = PRIVATE_DATA;

  // inc3
  {
    const int out_channels[] = { 96, 16, 64, 16, 32, 32 };

    setup_inception_sub(net, "inc3a", "conv3", out_channels, 2);
    setup_inception_sub(net, "inc3b", "inc3a", out_channels, 1);
    setup_inception_sub(net, "inc3c", "inc3b", out_channels, 1);
    setup_inception_sub(net, "inc3d", "inc3c", out_channels, 1);
    setup_inception_sub(net, "inc3e", "inc3d", out_channels, 1);
  }

  // inc4
  {
    const int out_channels[] = { 128, 32, 96, 16, 32, 32 };

    setup_inception_sub(net, "inc4a", "inc3e", out_channels, 2);
    setup_inception_sub(net, "inc4b", "inc4a", out_channels, 1);
    setup_inception_sub(net, "inc4c", "inc4b", out_channels, 1);
    setup_inception_sub(net, "inc4d", "inc4c", out_channels, 1);
    setup_inception_sub(net, "inc4e", "inc4d", out_channels, 1);
  }

  // hypercolumn
  {
    const int out_channels[] = { 256, 256 };

    setup_hyper_sub(net, "conv3", "inc3e", "inc4e",
                    out_channels);
  }
}

void setup_frcnn(Net* const net,
                 const char* const rpn_input_name,
                 const char* const rcnn_input_name,
                 const int rpn_channels,
                 const int rpn_kernel_h, const int rpn_kernel_w,
                 const int fc6_channels, const int fc7_channels)
{
  add_conv_layer(net, "rpn_conv1", rpn_input_name, "rpn_conv1", NULL, NULL,
                 1, rpn_channels, rpn_kernel_h, rpn_kernel_w, 1, 1,
                 (rpn_kernel_h - 1) / 2, (rpn_kernel_w - 1) / 2, 1, 0);
  add_relu_layer(net, "rpn_relu1", "rpn_conv1", "rpn_conv1", 0);

  add_conv_layer(net, "rpn_cls_score", "rpn_conv1", "rpn_cls_score",
                 NULL, NULL,
                 1, 50, 1, 1, 1, 1, 0, 0, 1, 0);
  get_layer_by_name(net, "rpn_cls_score")->f_forward[1]
      = forward_rpn_pred_layer;
  get_layer_by_name(net, "rpn_cls_score")->f_shape[1]
      = shape_rpn_pred_layer;

  add_conv_layer(net, "rpn_bbox_pred", "rpn_conv1", "rpn_bbox_pred",
                 NULL, NULL,
                 1, 100, 1, 1, 1, 1, 0, 0, 1, 0);
  get_layer_by_name(net, "rpn_bbox_pred")->f_forward[1]
      = forward_rpn_bbox_layer;
  get_layer_by_name(net, "rpn_bbox_pred")->f_shape[1]
      = shape_rpn_bbox_layer;

  // proposal, RoI-pooling
  {
    real anchor_scales[5] = { 3.0f, 6.0f, 9.0f, 16.0f, 32.0f };
    real anchor_ratios[5] = { 0.5f, 0.667f, 1.0f, 1.5f, 2.0f };
    Layer* const layer = add_layer(net, "proposal");
    layer->option.num_scales = 5;
    layer->option.num_ratios = 5;

    memcpy(net->anchor_scales, anchor_scales,
           layer->option.num_scales * sizeof(real));
    memcpy(net->anchor_ratios, anchor_ratios,
           layer->option.num_ratios * sizeof(real));

    layer->option.base_size = 16;
    layer->option.feat_stride = 16;
    layer->option.min_size = 16;
    layer->option.pre_nms_topn = 6000;
    layer->option.post_nms_topn = 300;
    layer->option.nms_thresh = 0.7f;
    layer->option.bbox_vote = 0;
    layer->option.vote_thresh = 0.7f;
    layer->option.scales = &net->anchor_scales[0];
    layer->option.ratios = &net->anchor_ratios[0];
    add_bottom(layer, get_tensor_by_name(net, "rpn_cls_score"));
    add_bottom(layer, get_tensor_by_name(net, "rpn_bbox_pred"));
    add_bottom(layer, get_tensor_by_name(net, "img_info"));
    add_top(layer, add_tensor(net, "rpn_roi"));
    layer->num_aux_data = 1;
    layer->f_forward[0] = forward_proposal_layer;
    layer->f_shape[0] = shape_proposal_layer;
    layer->f_init[0] = init_proposal_layer;
  }
  {
    Layer* const layer = add_layer(net, "rcnn_roipool");
    Tensor* const top = add_tensor(net, "rcnn_roipool");
    layer->option.pooled_height = 6;
    layer->option.pooled_width = 6;
    layer->option.spatial_scale = 0.0625;
    layer->option.flatten = 1;
    add_bottom(layer, get_tensor_by_name(net, rcnn_input_name));
    add_bottom(layer, get_tensor_by_name(net, "rpn_roi"));
    add_top(layer, top);
    top->data_type = PRIVATE_DATA;
    layer->f_forward[0] = forward_roipool_layer;
    layer->f_shape[0] = shape_roipool_layer;
  }

  add_fc_layer(net, "fc6_L", "rcnn_roipool", "fc6_L", NULL, NULL,
               fc6_channels, 0, 0);
  add_fc_layer(net, "fc6_U", "fc6_L", "fc6_U", NULL, NULL,
               4096, 1, 0);
  add_relu_layer(net, "relu6", "fc6_U", "fc6_U", 0);
  add_dropout_layer(net, "drop6", "fc6_U", "fc6_U", 1, 1);

  add_fc_layer(net, "fc7_L", "fc6_U", "fc7_L", NULL, NULL,
               fc7_channels, 0, 0);
  add_fc_layer(net, "fc7_U", "fc7_L", "fc7_U", NULL, NULL,
               4096, 1, 0);
  add_relu_layer(net, "relu7", "fc7_U", "fc7_U", 0);
  add_dropout_layer(net, "drop7", "fc7_U", "fc7_U", 1, 1);

  add_fc_layer(net, "cls_score", "fc7_U", "cls_score", NULL, NULL,
               21, 1, 0);
  {
    Layer* const layer = add_layer(net, "cls_pred");
    add_bottom(layer, get_tensor_by_name(net, "cls_score"));
    add_bottom(layer, get_tensor_by_name(net, "rpn_roi"));
    add_top(layer, add_tensor(net, "cls_pred"));
    get_layer_by_name(net, "cls_pred")->f_forward[0]
        = forward_rcnn_pred_layer;
    get_layer_by_name(net, "cls_pred")->f_shape[0]
        = shape_rcnn_pred_layer;
  }
  {
    Layer* const layer = add_fc_layer(net, "bbox_pred",
                 "fc7_U", "bbox_pred", NULL, NULL, 84, 1, 0);
    add_bottom(layer, get_tensor_by_name(net, "rpn_roi"));
    get_layer_by_name(net, "bbox_pred")->f_forward[1]
        = forward_rcnn_bbox_layer;
    get_layer_by_name(net, "bbox_pred")->f_shape[1]
        = shape_rcnn_bbox_layer;
  }
  {
    Layer* const layer = add_layer(net, "out");
    layer->option.min_size = 16;
    layer->option.score_thresh = 0.7f;
    layer->option.nms_thresh = 0.4f;
    layer->option.bbox_vote = 1;
    layer->option.vote_thresh = 0.5f;
    add_bottom(layer, get_tensor_by_name(net, "cls_pred"));
    add_bottom(layer, get_tensor_by_name(net, "bbox_pred"));
    add_bottom(layer, get_tensor_by_name(net, "rpn_roi"));
    add_bottom(layer, get_tensor_by_name(net, "img_info"));
    add_top(layer, add_tensor(net, "out"));
    layer->f_forward[0] = forward_odout_layer;
    layer->f_shape[0] = shape_odout_layer;
  }
  #ifndef DEMO
  {
    Layer* const layer = add_layer(net, "test");
    add_bottom(layer, get_tensor_by_name(net, "cls_pred"));
    add_bottom(layer, get_tensor_by_name(net, "bbox_pred"));
    add_bottom(layer, get_tensor_by_name(net, "rpn_roi"));
    add_bottom(layer, get_tensor_by_name(net, "img_info"));
    add_top(layer, add_tensor(net, "test"));
    layer->f_forward[0] = forward_odtest_layer;
    layer->f_shape[0] = shape_odtest_layer;
  }
  #endif
}

void construct_pvanet(Net* const pvanet,
                      const char* const param_path)
{
  init_net(pvanet);

  strcpy(pvanet->param_path, param_path);

  setup_shared_conv_sub(pvanet);
  //setup_frcnn(pvanet, "convf", "convf", 256, 1, 1, 512, 128);
  setup_frcnn(pvanet, "convf_rpn", "convf", 384, 3, 3, 512, 512);

  shape_net(pvanet);

  printf("Max layer size = %ld\n", pvanet->layer_size);
  printf("Max param size = %ld\n", pvanet->param_size);
  printf("Max temp size = %ld\n", pvanet->temp_size);
  printf("Max tempint size = %ld\n", pvanet->tempint_size);
  printf("Max const size = %ld\n", pvanet->const_size);

  malloc_net(pvanet);

  init_layers(pvanet);

  // print total memory size required
  {
  #ifdef GPU
    printf("%ldMB of main memory allocated\n",
           DIV_THEN_CEIL(pvanet->space_cpu,  1000000));
    printf("%ldMB of GPU memory allocated\n",
           DIV_THEN_CEIL(pvanet->space,  1000000));
  #else
    printf("%ldMB of main memory allocated\n",
           DIV_THEN_CEIL(pvanet->space + pvanet->space_cpu,  1000000));
  #endif
  }
}

void set_input_pvanet(Net* const net,
                      const unsigned char* const images_data[],
                      const int heights[],
                      const int widths[],
                      const int num_images)
{

  Tensor* const input = get_tensor_by_name(net, "data");
  Tensor* const img_info = get_tensor_by_name(net, "img_info");
  int shape_changed = (input->num_items != num_images);

  if (!shape_changed) {
    for (int n = 0; n < num_images; ++n) {
      if (img_info->data[n * 6 + 4] != (real)heights[n] ||
          img_info->data[n * 6 + 5] != (real)widths[n])
      {
        shape_changed = 1;
        break;
      }
    }
  }

  input->ndim = 3;
  input->num_items = 0;
  input->start[0] = 0;

  img_info->ndim = 1;
  img_info->num_items = 0;

  for (int n = 0; n < num_images; ++n) {
    img2input(images_data[n], input, img_info,
              (unsigned char*)net->temp_data,
              heights[n], widths[n]);
  }

  if (shape_changed) {
    printf("shape changed\n");
    shape_net(net);
  }
}

void get_output_pvanet(Net* const net,
                       const int image_start_index,
                       FILE* fp)
{
  // retrieve & save test output for measuring performance
  #ifndef DEMO
  {
    const Tensor* const out = get_tensor_by_name(net, "test");
    const long int output_size = flatten_size(out);

  #ifdef GPU
    cudaMemcpyAsync(net->temp_cpu_data, out->data,
                    output_size * sizeof(real),
                    cudaMemcpyDeviceToHost);
  #else
    memcpy(net->temp_cpu_data, out->data, output_size * sizeof(real));
  #endif

    if (fp) {
      for (int n = 0; n < out->num_items; ++n) {
        const real* const p_out_item = net->temp_cpu_data + out->start[n];

        fwrite(&out->ndim, sizeof(int), 1, fp);
        fwrite(out->shape[n], sizeof(int), out->ndim, fp);
        fwrite(p_out_item, sizeof(real), out->shape[n][0] * 6, fp);
      }
    }
  }
  #endif

  // retrieve & print output
  {
    const Tensor* const out = get_tensor_by_name(net, "out");
    const long int output_size = flatten_size(out);

  #ifdef GPU
    cudaMemcpyAsync(net->temp_cpu_data, out->data,
                    output_size * sizeof(real),
                    cudaMemcpyDeviceToHost);
  #else
    memcpy(net->temp_cpu_data, out->data, output_size * sizeof(real));
  #endif

    net->num_output_boxes = 0;
    for (int n = 0; n < out->num_items; ++n) {
      const int image_index = image_start_index + n;
      const real* const p_out_item = net->temp_cpu_data + out->start[n];

      for (int i = 0; i < out->shape[n][0]; ++i) {
        const int class_index = (int)p_out_item[i * 6 + 0];

        printf("Image %d / Box %d: ", image_index, i);
        printf("class %d, score %f, p1 = (%.2f, %.2f), p2 = (%.2f, %.2f)\n",
               class_index, p_out_item[i * 6 + 5],
               p_out_item[i * 6 + 1], p_out_item[i * 6 + 2],
               p_out_item[i * 6 + 3], p_out_item[i * 6 + 4]);
      }
      net->num_output_boxes += out->shape[n][0];
    }
  }
}

void process_pvanet(Net* const net,
                    const unsigned char image_data[],
                    const int height,
                    const int width,
                    FILE* fp)
{
  set_input_pvanet(net, &image_data, &height, &width, 1);

  forward_net(net);

  get_output_pvanet(net, 0, fp);
}

void process_batch_pvanet(Net* const net,
                          const unsigned char* const images_data[],
                          const int heights[],
                          const int widths[],
                          const int num_images,
                          FILE* fp)
{
  set_input_pvanet(net, images_data, heights, widths, num_images);

  forward_net(net);

  get_output_pvanet(net, 0, fp);
}
