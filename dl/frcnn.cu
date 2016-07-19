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
  layer->option.out_channels = num_output;
  layer->option.num_groups = num_group;
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

  layer->option.out_channels = num_output;
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
                         const int is_test,
                         const int is_scaled)
{
  Layer* const layer =
      add_chain_layer(net, layer_name, bottom_name, top_name);

  layer->option.test = is_test;
  layer->option.scaled = is_scaled;

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
                          real anchor_scales[], const int num_scales,
                          real anchor_ratios[], const int num_ratios,
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

  layer->option.num_scales = num_scales;
  layer->option.num_ratios = num_ratios;
  layer->option.base_size = base_size;
  layer->option.feat_stride = feat_stride;
  layer->option.min_size = min_size;
  layer->option.pre_nms_topn = pre_nms_topn;
  layer->option.post_nms_topn = post_nms_topn;
  layer->option.nms_thresh = nms_thresh;
  layer->option.bbox_vote = bbox_vote;
  layer->option.vote_thresh = vote_thresh;
  layer->option.scales = anchor_scales;
  layer->option.ratios = anchor_ratios;

  layer->f_forward = forward_proposal_layer;
  layer->f_shape = shape_proposal_layer;
  layer->f_free = free_proposal_layer;

  malloc_proposal_layer(net, layer);

  return layer;
}

void setup_frcnn(Net* const net,
                 const char* const rpn_input_name,
                 const char* const rcnn_input_name,
                 const int rpn_channels,
                 const int rpn_kernel_h, const int rpn_kernel_w,
                 const int fc_compress,
                 const int fc6_dim, const int fc7_dim,
                 const int pre_nms_topn, const int post_nms_topn)
{
  add_conv_layer(net, "rpn_conv1", rpn_input_name, "rpn_conv1", NULL, NULL,
                 1, rpn_channels, rpn_kernel_h, rpn_kernel_w, 1, 1,
                 (rpn_kernel_h - 1) / 2, (rpn_kernel_w - 1) / 2, 1);
  add_relu_layer(net, "rpn_relu1", "rpn_conv1", "rpn_conv1", 0);

  add_conv_layer(net, "rpn_cls_score", "rpn_conv1", "rpn_cls_score",
                 NULL, NULL,
                 1, 50, 1, 1, 1, 1, 0, 0, 1);

  {
    //int rpn_score_shape[] = { 2, 0, 0, -1 };
    int rpn_score_shape[] = { 2, -1, 0 };
    add_reshape_layer(net, "rpn_cls_score_reshape",
                      "rpn_cls_score", "rpn_cls_score", rpn_score_shape, 3);
  }

  add_softmax_layer(net, "rpn_cls_pred", "rpn_cls_score", "rpn_cls_score",
                    0);

  {
    //int rpn_pred_shape[] = { -1, 0, 0 };
    int rpn_pred_shape[] = { 50, -1, 0 };
    add_reshape_layer(net, "rpn_cls_pred_reshape",
                      "rpn_cls_score", "rpn_cls_score", rpn_pred_shape, 3);
  }

  add_conv_layer(net, "rpn_bbox_pred", "rpn_conv1", "rpn_bbox_pred",
                 NULL, NULL,
                 1, 100, 1, 1, 1, 1, 0, 0, 1);

  // proposal, RoI-pooling
  {
    real anchor_scales[5] = { 3.0f, 6.0f, 9.0f, 16.0f, 32.0f };
    real anchor_ratios[5] = { 0.5f, 0.667f, 1.0f, 1.5f, 2.0f };
    add_proposal_layer(net, "proposal", "rpn_cls_score", "rpn_bbox_pred", "img_info", "rpn_roi",
                       anchor_scales, 5, anchor_ratios, 5, 16, 16, 16, pre_nms_topn, post_nms_topn, 0.7f, 0, 0.7f);
  }
  {
    Layer* const layer = add_layer(net, "rcnn_roipool");
    Tensor* const top = add_tensor(net, "rcnn_roipool");
    top->data_type = PRIVATE_DATA;
    layer->option.pooled_height = 6;
    layer->option.pooled_width = 6;
    layer->option.spatial_scale = 0.0625;
    layer->option.flatten = 1;
    add_bottom(layer, get_tensor_by_name(net, rcnn_input_name));
    add_bottom(layer, get_tensor_by_name(net, "rpn_roi"));
    add_top(layer, top);
    layer->f_forward = forward_roipool_layer;
    layer->f_shape = shape_roipool_layer;
  }

  if (fc_compress) {
    add_fc_layer(net, "fc6_L", "rcnn_roipool", "fc6_L", NULL, NULL,
                 fc6_dim, 0);
    add_fc_layer(net, "fc6_U", "fc6_L", "fc6_U", NULL, NULL, 4096, 1);
    add_relu_layer(net, "relu6", "fc6_U", "fc6_U", 0);
    add_dropout_layer(net, "drop6", "fc6_U", "fc6_U", 1, 1);

    add_fc_layer(net, "fc7_L", "fc6_U", "fc7_L", NULL, NULL, fc7_dim, 0);
    add_fc_layer(net, "fc7_U", "fc7_L", "fc7_U", NULL, NULL, 4096, 1);
    add_relu_layer(net, "relu7", "fc7_U", "fc7_U", 0);
    add_dropout_layer(net, "drop7", "fc7_U", "fc7_U", 1, 1);

    add_fc_layer(net, "cls_score", "fc7_U", "cls_score", NULL, NULL, 21, 1);
    add_fc_layer(net, "bbox_pred", "fc7_U", "bbox_pred", NULL, NULL, 84, 1);
  }
  else {
    add_fc_layer(net, "fc6", "rcnn_roipool", "fc6", NULL, NULL, 4096, 1);
    add_relu_layer(net, "relu6", "fc6", "fc6", 0);
    add_dropout_layer(net, "drop6", "fc6", "fc6", 1, 1);

    add_fc_layer(net, "fc7", "fc6", "fc7", NULL, NULL, 4096, 1);
    add_relu_layer(net, "relu7", "fc7", "fc7", 0);
    add_dropout_layer(net, "drop7", "fc7", "fc7", 1, 1);

    add_fc_layer(net, "cls_score", "fc7", "cls_score", NULL, NULL, 21, 1);
    add_fc_layer(net, "bbox_pred", "fc7", "bbox_pred", NULL, NULL, 84, 1);
  }

  add_softmax_layer(net, "cls_pred", "cls_score", "cls_score", 1);

  {
    Layer* const layer = add_layer(net, "out");
    layer->option.min_size = 16;
    layer->option.pre_nms_topn = post_nms_topn;
    layer->option.score_thresh = 0.7f;
    layer->option.nms_thresh = 0.4f;
    layer->option.bbox_vote = 0;
    layer->option.vote_thresh = 0.5f;
    add_bottom(layer, get_tensor_by_name(net, "cls_score"));
    add_bottom(layer, get_tensor_by_name(net, "bbox_pred"));
    add_bottom(layer, get_tensor_by_name(net, "rpn_roi"));
    add_bottom(layer, get_tensor_by_name(net, "img_info"));
    add_top(layer, add_tensor(net, "out"));
    layer->f_forward = forward_odout_layer;
    layer->f_shape = shape_odout_layer;
    layer->f_free = free_odout_layer;
    malloc_odout_layer(net, layer);
  }
}

void construct_pvanet(Net* const pvanet,
                      const char* const param_path,
                      const int is_light_model,
                      const int fc_compress,
                      const int pre_nms_topn,
                      const int post_nms_topn,
                      const int input_scale)
{
  init_net(pvanet);

  strcpy(pvanet->param_path, param_path);
  pvanet->input_scale = input_scale;

  if (is_light_model) {
    setup_shared_cnn_light(pvanet);
    setup_frcnn(pvanet, "convf", "convf", 256, 1, 1,
                fc_compress, 512, 128, pre_nms_topn, post_nms_topn);
  }
  else {
    setup_shared_cnn(pvanet);
    setup_frcnn(pvanet, "convf_rpn", "convf", 384, 3, 3,
                fc_compress, 512, 512, pre_nms_topn, post_nms_topn);
  }

  shape_net(pvanet);

  malloc_net(pvanet);
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
    #ifdef GPU
    real img_info_cpu[BATCH_SIZE * 6];
    const real* const p_img_info_cpu = img_info_cpu;
    cudaMemcpyAsync(img_info_cpu, img_info->data,
                    get_data_size(img_info) * sizeof(real),
                    cudaMemcpyDeviceToHost);
    #else
    const real* const p_img_info_cpu = img_info->data;
    #endif
    for (int n = 0; n < num_images; ++n) {
      if (p_img_info_cpu[n * 6 + 4] != (real)heights[n] ||
          p_img_info_cpu[n * 6 + 5] != (real)widths[n])
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
              heights[n], widths[n], net->input_scale);
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
  const Tensor* const out = get_tensor_by_name(net, "out");
  const long int output_size = get_data_size(out);

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

  net->num_output_boxes = 0;
  for (int n = 0; n < out->num_items; ++n) {
    const int image_index = image_start_index + n;
    const real* const p_out_item = net->temp_cpu_data + out->start[n];

    for (int i = 0; i < out->shape[n][0]; ++i) {
      const int class_index = (int)p_out_item[i * 6 + 0];

      if (p_out_item[i * 6 + 5] < 0.7f) {
        continue;
      }
      printf("Image %d / Box %d: ", image_index, i);
      printf("class %d, score %f, p1 = (%.2f, %.2f), p2 = (%.2f, %.2f)\n",
             class_index, p_out_item[i * 6 + 5],
             p_out_item[i * 6 + 1], p_out_item[i * 6 + 2],
             p_out_item[i * 6 + 3], p_out_item[i * 6 + 4]);
    }
    net->num_output_boxes += out->shape[n][0];
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
