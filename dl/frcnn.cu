#include "layer.h"
#include <string.h>

//#define MSRPN
#define FC_COMPRESS
#define INCEPTION
//#define INCEPTION64

#define DROPOUT_SCALE_TRAIN 1  // for new PVANET
//#define DROPOUT_SCALE_TRAIN 0  // for old PVANET

void setup_data_layer(Net* const net,
                      const char* const layer_name,
                      const char* const top_name)
{
  Layer* const layer = add_layer(net, layer_name);

  add_top(layer, add_tensor(net, top_name));

  input_init_shape(net, get_top(layer, 0), &net->img_info);
}

void setup_conv_layer(Net* const net,
                     const char* const layer_name,
                     const char* const bottom_name,
                     const char* const top_name,
                     const char* const weight_name,
                     const char* const bias_name,
                     const int out_channels,
                     const int kernel,
                     const int stride,
                     const int add_bias,
                     const int do_relu)
{
  Layer* const layer = add_layer(net, layer_name);

  add_bottom(layer, get_tensor_by_name(net, bottom_name));
  add_top(layer, add_tensor(net, top_name));

  if (!weight_name) {
    char temp_name[64];
    sprintf(temp_name, "%s_param%d", layer_name, layer->num_params);
    add_param(layer, add_tensor(net, temp_name));
  }
  else {
    add_param(layer, find_or_add_tensor(net, weight_name));
  }

  if (add_bias) {
    if (!bias_name) {
      char temp_name[64];
      sprintf(temp_name, "%s_param%d", layer_name, layer->num_params);
      add_param(layer, add_tensor(net, temp_name));
    }
    else {
      add_param(layer, find_or_add_tensor(net, bias_name));
    }
  }

  layer->option.kernel_h = kernel;
  layer->option.kernel_w = kernel;
  layer->option.stride_h = stride;
  layer->option.stride_w = stride;
  layer->option.pad_h = (kernel - 1) / 2;
  layer->option.pad_w = (kernel - 1) / 2;
  layer->option.out_channels = out_channels;
  layer->option.num_groups = 1;
  layer->option.bias = add_bias;
  layer->option.handle = (void*)&net->blas_handle;

  layer->f_forward[0] = forward_conv_layer;
  if (do_relu) {
    layer->f_forward[1] = forward_inplace_relu_layer;
  }
  layer->f_shape[0] = shape_conv_layer;
}

static
void setup_inception_sub(Net* const net,
                         const char* const sub_name,
                         const char* const input_name,
                         const int* const out_channels,
                         const int stride)
{
  char bottom_name[64];
  char top_name[64];
  Layer* layer;

  if (stride > 1) {
    // pool1
    sprintf(top_name, "%s_pool1", sub_name);
    layer = add_layer(net, top_name);
    add_bottom(layer, get_tensor_by_name(net, input_name));
    add_top(layer, add_tensor(net, top_name));
    layer->option.kernel_h = 3;
    layer->option.kernel_w = 3;
    layer->option.stride_h = stride;
    layer->option.stride_w = stride;
    layer->option.pad_h = 0;
    layer->option.pad_w = 0;
    layer->f_forward[0] = forward_pool_layer;
    layer->f_shape[0] = shape_pool_layer;

    // conv1
    sprintf(bottom_name, "%s_pool1", sub_name);
    sprintf(top_name, "%s_conv1", sub_name);
    setup_conv_layer(net, top_name,
                     bottom_name, top_name, NULL, NULL,
                     out_channels[0], 1, 1, 1, 1);
  }
  else {
    // conv1
    sprintf(top_name, "%s_conv1", sub_name);
    setup_conv_layer(net, top_name,
                     input_name, top_name, NULL, NULL,
                     out_channels[0], 1, 1, 1, 1);
  }

  // conv3_1
  sprintf(top_name, "%s_conv3_1", sub_name);
  setup_conv_layer(net, top_name,
                   input_name, top_name, NULL, NULL,
                   out_channels[1], 1, 1, 1, 1);

  // conv3_2
  sprintf(bottom_name, "%s_conv3_1", sub_name);
  sprintf(top_name, "%s_conv3_2", sub_name);
  setup_conv_layer(net, top_name,
                   bottom_name, top_name, NULL, NULL,
                   out_channels[2], 3, stride, 1, 1);

  // conv5_1
  sprintf(top_name, "%s_conv5_1", sub_name);
  setup_conv_layer(net, top_name,
                   input_name, top_name, NULL, NULL,
                   out_channels[3], 1, 1, 1, 1);

  // conv5_2
  sprintf(bottom_name, "%s_conv5_1", sub_name);
  sprintf(top_name, "%s_conv5_2", sub_name);
  setup_conv_layer(net, top_name,
                   bottom_name, top_name, NULL, NULL,
                   out_channels[4], 3, 1, 1, 1);

  // conv5_2
  sprintf(bottom_name, "%s_conv5_2", sub_name);
  sprintf(top_name, "%s_conv5_3", sub_name);
  setup_conv_layer(net, top_name,
                   bottom_name, top_name, NULL, NULL,
                   out_channels[5], 3, stride, 1, 1);

  // concat
  {
    sprintf(top_name, "%s_concat", sub_name);
    layer = add_layer(net, top_name);

    sprintf(bottom_name, "%s_conv1", sub_name);
    add_bottom(layer, get_tensor_by_name(net, bottom_name));
    sprintf(bottom_name, "%s_conv3_2", sub_name);
    add_bottom(layer, get_tensor_by_name(net, bottom_name));
    sprintf(bottom_name, "%s_conv5_3", sub_name);
    add_bottom(layer, get_tensor_by_name(net, bottom_name));

    add_top(layer, add_tensor(net, top_name));
    layer->option.num_concats = layer->num_bottoms;
    layer->f_forward[0] = forward_concat_layer;
    layer->f_shape[0] = shape_concat_layer;
  }
}

void setup_hyper_sub(Net* const net,
                     const char* const downsample_name,
                     const char* const as_is_name,
                     const char* const upsample_name,
                     const int* const out_channels)
{
  // downsample
  {
    const char* const bottom_name = downsample_name;
    const char* const top_name = "downsample";
    Layer* const layer = add_layer(net, top_name);

    add_bottom(layer, get_tensor_by_name(net, bottom_name));
    add_top(layer, add_tensor(net, top_name));

    layer->option.kernel_h = 3;
    layer->option.kernel_w = 3;
    layer->option.stride_h = 2;
    layer->option.stride_w = 2;
    layer->option.pad_h = 0;
    layer->option.pad_w = 0;
    layer->f_forward[0] = forward_pool_layer;
    layer->f_shape[0] = shape_pool_layer;
  }

  // upsample
  {
    const char* const bottom_name = upsample_name;
    const char* const top_name = "upsample";
    Layer* const layer = add_layer(net, top_name);

    add_bottom(layer, get_tensor_by_name(net, bottom_name));
    add_top(layer, add_tensor(net, top_name));
    {
      char param_name[64];
      sprintf(param_name, "%s_param%d", top_name, layer->num_params);
      add_param(layer, add_tensor(net, param_name));
    }

    layer->option.kernel_h = 4;
    layer->option.kernel_w = 4;
    layer->option.stride_h = 2;
    layer->option.stride_w = 2;
    layer->option.pad_h = 1;
    layer->option.pad_w = 1;
    layer->option.out_channels = out_channels[0];
    layer->option.num_groups = out_channels[0];
    layer->option.handle = (void*)&net->blas_handle;
    layer->f_forward[0] = forward_deconv_layer;
    layer->f_shape[0] = shape_deconv_layer;
  }

  // concat
  {
    const char* const top_name = "concat";
    Layer* const layer = add_layer(net, top_name);

    add_bottom(layer, get_tensor_by_name(net, "downsample"));
    add_bottom(layer, get_tensor_by_name(net, as_is_name));
    add_bottom(layer, get_tensor_by_name(net, "upsample"));
    add_top(layer, add_tensor(net, top_name));

    layer->option.num_concats = layer->num_bottoms;
    layer->f_forward[0] = forward_concat_layer;
    layer->f_shape[0] = shape_concat_layer;
  }

  // convf
  {
    const char* const bottom_name = "concat";
    const char* const top_name = "convf";
    setup_conv_layer(net, top_name,
                     bottom_name, top_name, NULL, NULL,
                     out_channels[1], 1, 1, 1, 1);
  }
}

static
void setup_inception(Net* const net)
{
  // data
  setup_data_layer(net, "data", "data");

  // conv1, conv2, conv3
  {
    const int sub_size = 3;
    const char* conv_names[] = {
      "conv1", "conv2", "conv3"
    };
  #ifdef INCEPTION64
    const int out_channels[] = { 32, 48, 96 };
    const int kernels[] = { 4, 3, 3 };
  #else
    const int out_channels[] = { 24, 48, 96 };
    const int kernels[] = { 7, 3, 3 };
  #endif
    const int strides[] = { 2, 2, 2 };

    for (int i = 0; i < sub_size; ++i) {
      const char* const bottom_name = (i > 0) ? conv_names[i - 1] : "data";
      const char* const top_name = conv_names[i];
      setup_conv_layer(net, conv_names[i],
                       bottom_name, top_name, NULL, NULL,
                       out_channels[i], kernels[i], strides[i], 1, 1);
    }

    get_tensor_by_name(net, "conv1")->has_own_memory = 1;
    get_tensor_by_name(net, "conv3")->has_own_memory = 1;
  }

  // inc3
  {
  #ifdef INCEPTION64
    const int out_channels[] = { 96, 16, 64, 16, 32, 32 };
  #else
    const int out_channels[] = { 96, 24, 64, 12, 24, 24 };
  #endif

    setup_inception_sub(net, "inc3a", "conv3", out_channels, 2);
    setup_inception_sub(net, "inc3b", "inc3a_concat", out_channels, 1);
    setup_inception_sub(net, "inc3c", "inc3b_concat", out_channels, 1);
    setup_inception_sub(net, "inc3d", "inc3c_concat", out_channels, 1);
    setup_inception_sub(net, "inc3e", "inc3d_concat", out_channels, 1);
  }

  // inc4
  {
    const int out_channels[] = { 128, 32, 96, 16, 32, 32 };

    setup_inception_sub(net, "inc4a", "inc3e_concat", out_channels, 2);
    setup_inception_sub(net, "inc4b", "inc4a_concat", out_channels, 1);
    setup_inception_sub(net, "inc4c", "inc4b_concat", out_channels, 1);
    setup_inception_sub(net, "inc4d", "inc4c_concat", out_channels, 1);
    setup_inception_sub(net, "inc4e", "inc4d_concat", out_channels, 1);
  }

  // hypercolumn
  {
    const int out_channels[] = { 256, 256 };

    setup_hyper_sub(net, "conv3", "inc3e_concat", "inc4e_concat",
                    out_channels);
  }

  net->num_layer_data = 5;
}

static
void setup_frcnn(Net* const net,
                 Layer* const layers,
                 const int rpn_channels,
                 Layer* const convnet_out_layer)
{
  const char* names[] = {
    // RPN
    #ifdef MSRPN
      "rpn_conv1", "rpn_cls_score1", "rpn_bbox_pred1",
      "rpn_conv3", "rpn_cls_score3", "rpn_bbox_pred3",
      "rpn_conv5", "rpn_cls_score5", "rpn_bbox_pred5",
      "rpn_score", "rpn_bbox",
    #else
      "rpn_conv1", "rpn_cls_score", "rpn_bbox_pred",
      "null", "null", "null",
      "null", "null", "null",
      "null", "null",
    #endif
      "rpn_roi",

    // R-CNN
      "rcnn_roipool",
    #ifdef FC_COMPRESS
      "fc6_L", "fc6_U", "fc7_L", "fc7_U",
    #else
      "null", "fc6", "null", "fc7",
    #endif
      "cls_score", "cls_pred", "bbox_pred",

    // output & test
    #ifdef DEMO
      "out"
    #else
      "out", "test"
    #endif
  };

  #ifdef DEMO
  const int sub_size = 21;
  #else
  const int sub_size = 22;
  #endif

  char temp_name[64];

  for (int i = 0; i < sub_size; ++i) {
    set_layer_name(&layers[i], names[i]);
  }

  {
    #ifdef MSRPN
    const int num_conv_layers = 9;
    #else
    const int num_conv_layers = 3;
    #endif

    for (int i = 0; i < num_conv_layers; ++i) {
      layers[i].option.kernel_h = 1;
      layers[i].option.kernel_w = 1;
      layers[i].option.stride_h = 1;
      layers[i].option.stride_w = 1;
      layers[i].option.pad_h = 0;
      layers[i].option.pad_w = 0;
      layers[i].option.num_groups = 1;
      layers[i].option.bias = 1;
      layers[i].option.negative_slope = 0;
      #ifdef GPU
      layers[i].option.handle = (void*)&net->blas_handle;
      #endif

      layers[i].num_bottoms = 1;
      add_top(&layers[i], add_tensor(net, names[i]));
      sprintf(temp_name, "%s_param%d", names[i], 0);
      add_param(&layers[i], add_tensor(net, temp_name));
      sprintf(temp_name, "%s_param%d", names[i], 1);
      add_param(&layers[i], add_tensor(net, temp_name));
    }
  }

  #ifdef MSRPN
  {
    // 1x1 RPN
    layers[0].option.out_channels = rpn_channels / 4;
    layers[1].option.out_channels = 18;
    layers[2].option.out_channels = 36;

    // 3x3 RPN
    layers[3].option.kernel_h = 3;
    layers[3].option.kernel_w = 3;
    layers[3].option.pad_h = 1;
    layers[3].option.pad_w = 1;
    layers[3].option.out_channels = rpn_channels / 2;
    layers[4].option.out_channels = 18;
    layers[5].option.out_channels = 36;

    // 5x5 RPN
    layers[6].option.kernel_h = 5;
    layers[6].option.kernel_w = 5;
    layers[6].option.pad_h = 2;
    layers[6].option.pad_w = 2;
    layers[6].option.out_channels = rpn_channels / 4;
    layers[7].option.out_channels = 18;
    layers[8].option.out_channels = 36;

    // score concat
    layers[9].option.num_concats = 3;
    layers[9].num_bottoms = 3;
    layers[9].num_tops = 1;

    // bbox concat
    layers[10].option.num_concats = 3;
    layers[10].num_bottoms = 3;
    layers[10].num_tops = 1;
  }
  #else
  {
    // 3x3 RPN if using PVA-7.1.1
    #ifndef INCEPTION
    layers[0].option.kernel_h = 3;
    layers[0].option.kernel_w = 3;
    layers[0].option.pad_h = 1;
    layers[0].option.pad_w = 1;
    #endif

    layers[0].option.out_channels = rpn_channels;
    layers[1].option.out_channels = 50;
    layers[2].option.out_channels = 100;
  }
  #endif

  // proposal, RoI-pooling
  {
  #ifdef MSRPN
    real anchor_scales[9] = { 3.0f, 6.0f, 9.0f,
                              4.0f, 8.0f, 16.0f,
                              7.0f, 13.0f, 32.0f };
    real anchor_ratios[3] = { 0.5f, 1.0f, 2.0f };
    layers[11].option.num_scales = 9;
    layers[11].option.num_ratios = 3;
  #else
    real anchor_scales[5] = { 3.0f, 6.0f, 9.0f, 16.0f, 32.0f };
    real anchor_ratios[5] = { 0.5f, 0.667f, 1.0f, 1.5f, 2.0f };
    layers[11].option.num_scales = 5;
    layers[11].option.num_ratios = 5;
  #endif

    memcpy(net->anchor_scales, anchor_scales,
           layers[11].option.num_scales * sizeof(real));
    memcpy(net->anchor_ratios, anchor_ratios,
           layers[11].option.num_ratios * sizeof(real));

    layers[11].option.num_concats = 1;
    layers[11].option.base_size = 16;
    layers[11].option.feat_stride = 16;
    layers[11].option.min_size = 16;
    layers[11].option.pre_nms_topn = 6000;
    layers[11].option.post_nms_topn = 300;
    layers[11].option.nms_thresh = 0.7f;
    layers[11].option.scales = &net->anchor_scales[0];
    layers[11].option.ratios = &net->anchor_ratios[0];
    layers[11].num_bottoms = 3;
    add_top(&layers[11], add_tensor(net, names[11]));
    layers[11].num_aux_data = 1;

    layers[12].option.pooled_height = 6;
    layers[12].option.pooled_width = 6;
    layers[12].option.spatial_scale = 0.0625;
    layers[12].option.flatten = 1;
    layers[12].num_bottoms = 2;
    add_top(&layers[12], add_tensor(net, names[12]));
    layers[12].p_tops[0]->has_own_memory = 1;
  }

  // fc6, fc7, RCNN score, RCNN bbox
  for (int i = 13; i <= 19; ++i) {
    layers[i].option.bias = 1;
    layers[i].option.negative_slope = 0;
    layers[i].option.threshold = 0.5f;
    layers[i].option.test = 1;
    layers[i].option.scaled = DROPOUT_SCALE_TRAIN;
    #ifdef GPU
    layers[i].option.handle = (void*)&net->blas_handle;
    #endif

    layers[i].num_bottoms = 1;
    if (i != 13 && i != 15 && i != 18) {
      add_top(&layers[i], add_tensor(net, names[i]));
      sprintf(temp_name, "%s_param%d", names[i], 0);
      add_param(&layers[i], add_tensor(net, temp_name));
      sprintf(temp_name, "%s_param%d", names[i], 1);
      add_param(&layers[i], add_tensor(net, temp_name));
    }
  }

  // fc6, fc7
  {
  #ifdef FC_COMPRESS
    layers[13].option.out_channels = 512;
    layers[13].option.bias = 0;
    add_top(&layers[13], add_tensor(net, names[13]));
    sprintf(temp_name, "%s_param%d", names[13], 0);
    add_param(&layers[13], add_tensor(net, temp_name));

    layers[15].option.out_channels = 128;
    layers[15].option.bias = 0;
    add_top(&layers[15], add_tensor(net, names[15]));
    sprintf(temp_name, "%s_param%d", names[15], 0);
    add_param(&layers[15], add_tensor(net, temp_name));
  #else
    layers[13].num_bottoms = 0;
    layers[15].num_bottoms = 0;
  #endif
    layers[14].option.out_channels = 4096;
    layers[16].option.out_channels = 4096;
  }

  // RCNN score
  {
  #ifdef DEMO
    layers[17].option.out_channels = 25;
  #else
    layers[17].option.out_channels = 21;
  #endif
  }

  // RCNN pred
  layers[18].num_bottoms = 2;
  add_top(&layers[18], add_tensor(net, names[18]));

  // RCNN bbox
  layers[19].option.out_channels = layers[17].option.out_channels * 4;
  layers[19].num_bottoms = 2;

  // output
  layers[20].option.min_size = 16;
  layers[20].option.score_thresh = 0.7f;
  layers[20].option.nms_thresh = 0.3f;
  layers[20].num_bottoms = 4;
  add_top(&layers[20], add_tensor(net, names[20]));

  // test
  #ifndef DEMO
  {
    layers[21].num_bottoms = 4;
    add_top(&layers[21], add_tensor(net, names[21]));
  }
  #endif

  // RPN
  {
    #ifdef MSRPN
    const int num_conv_layers = 9;
    #else
    const int num_conv_layers = 3;
    #endif

    for (int i = 0; i < num_conv_layers; i += 3) {
      // conv
      layers[i].p_bottoms[0] = convnet_out_layer->p_tops[0];
      layers[i].f_forward[0] = forward_conv_layer;
      layers[i].f_forward[1] = forward_inplace_relu_layer;
      layers[i].f_shape[0] = shape_conv_layer;

      // score
      layers[i + 1].p_bottoms[0] = layers[i].p_tops[0];
      layers[i + 1].f_forward[0] = forward_conv_layer;
      layers[i + 1].f_shape[0] = shape_conv_layer;

      // bbox
      layers[i + 2].p_bottoms[0] = layers[i].p_tops[0];
      layers[i + 2].f_forward[0] = forward_conv_layer;
      layers[i + 2].f_shape[0] = shape_conv_layer;
    }
  }

  #ifdef MSRPN
  {
    // score concat
    layers[9].p_bottoms[0] = layers[1].p_tops[0];
    layers[9].p_bottoms[1] = layers[4].p_tops[0];
    layers[9].p_bottoms[2] = layers[7].p_tops[0];
    layers[9].f_forward[0] = forward_concat_layer;
    layers[9].f_forward[1] = forward_rpn_pred_layer;
    layers[9].f_shape[0] = shape_concat_layer;
    layers[9].f_shape[1] = shape_rpn_pred_layer;

    // bbox concat
    layers[10].p_bottoms[0] = layers[2].p_tops[0];
    layers[10].p_bottoms[1] = layers[5].p_tops[0];
    layers[10].p_bottoms[2] = layers[8].p_tops[0];
    layers[10].f_forward[0] = forward_concat_layer;
    layers[10].f_forward[1] = forward_rpn_bbox_layer;
    layers[10].f_shape[0] = shape_concat_layer;
    layers[10].f_shape[1] = shape_rpn_bbox_layer;
  }
  #else
  {
    layers[1].f_forward[1] = forward_rpn_pred_layer;
    layers[1].f_shape[1] = shape_rpn_pred_layer;
    layers[2].f_forward[1] = forward_rpn_bbox_layer;
    layers[2].f_shape[1] = shape_rpn_bbox_layer;
  }
  #endif

  // proposal & RoI-pooling
  {
  #ifdef MSRPN
    layers[11].p_bottoms[0] = layers[9].p_tops[0];
    layers[11].p_bottoms[1] = layers[10].p_tops[0];
  #else
    layers[11].p_bottoms[0] = layers[1].p_tops[0];
    layers[11].p_bottoms[1] = layers[2].p_tops[0];
  #endif
    layers[11].p_bottoms[2] = &net->img_info;
    layers[11].f_forward[0] = forward_proposal_layer;
    layers[11].f_shape[0] = shape_proposal_layer;
    layers[11].f_init[0] = init_proposal_layer;

    layers[12].p_bottoms[0] = convnet_out_layer->p_tops[0];
    layers[12].p_bottoms[1] = layers[11].p_tops[0];
    layers[12].f_forward[0] = forward_roipool_layer;
    layers[12].f_shape[0] = shape_roipool_layer;
  }

  // fc6_L, 6_U, 7_L, 7_U
  for (int i = 13; i <= 16; i += 2) {
  #ifdef FC_COMPRESS
    layers[i].p_bottoms[0] = layers[i - 1].p_tops[0];
    layers[i].f_forward[0] = forward_fc_layer;
    layers[i].f_shape[0] = shape_fc_layer;
    layers[i + 1].p_bottoms[0] = layers[i].p_tops[0];
  #else
    layers[i + 1].p_bottoms[0] = layers[i - 1].p_tops[0];
  #endif
    layers[i + 1].f_forward[0] = forward_fc_layer;
    layers[i + 1].f_forward[1] = forward_inplace_relu_layer;
    layers[i + 1].f_forward[2] = forward_inplace_dropout_layer;
    layers[i + 1].f_shape[0] = shape_fc_layer;
  }

  // RCNN score
  layers[17].p_bottoms[0] = layers[16].p_tops[0];
  layers[17].f_forward[0] = forward_fc_layer;
  layers[17].f_shape[0] = shape_fc_layer;

  // RCNN pred
  layers[18].p_bottoms[0] = layers[17].p_tops[0];
  layers[18].p_bottoms[1] = layers[11].p_tops[0];
  layers[18].f_forward[0] = forward_rcnn_pred_layer;
  layers[18].f_shape[0] = shape_rcnn_pred_layer;

  // RCNN bbox
  layers[19].p_bottoms[0] = layers[16].p_tops[0];
  layers[19].p_bottoms[1] = layers[11].p_tops[0];
  layers[19].f_forward[0] = forward_fc_layer;
  layers[19].f_forward[1] = forward_rcnn_bbox_layer;
  layers[19].f_shape[0] = shape_fc_layer;
  layers[19].f_shape[1] = shape_rcnn_bbox_layer;

  // output
  layers[20].p_bottoms[0] = layers[18].p_tops[0];
  layers[20].p_bottoms[1] = layers[19].p_tops[0];
  layers[20].p_bottoms[2] = layers[11].p_tops[0];
  layers[20].p_bottoms[3] = &net->img_info;
  layers[20].f_forward[0] = forward_odout_layer;
  layers[20].f_shape[0] = shape_odout_layer;

  // test
  #ifndef DEMO
  {
    layers[21].p_bottoms[0] = layers[18].p_tops[0];
    layers[21].p_bottoms[1] = layers[19].p_tops[0];
    layers[21].p_bottoms[2] = layers[11].p_tops[0];
    layers[21].p_bottoms[3] = &net->img_info;
    layers[21].f_forward[0] = forward_odtest_layer;
    layers[21].f_shape[0] = shape_odtest_layer;
  }
  #endif

  net->num_layers += sub_size;
}

void construct_pvanet(Net* const pvanet,
                      const char* const param_path)
{
  init_net(pvanet);

  strcpy(pvanet->param_path, param_path);

  #ifdef INCEPTION
  setup_inception(pvanet);
  #else
  setup_pva711(pvanet);
  #endif
  setup_frcnn(pvanet, &pvanet->layers[pvanet->num_layers],
              pvanet->layers[pvanet->num_layers - 1].option.out_channels,
              &pvanet->layers[pvanet->num_layers - 1]);

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
                      const unsigned char* const * const images_data,
                      const int* const heights,
                      const int* const widths,
                      const int num_images)
{

  Tensor* const input = net->layers[0].p_tops[0];
  Tensor* const img_info = &net->img_info;
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
    print_tensor_info("data", input);
    print_tensor_info("img_info", img_info);
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
    const Tensor* const out = net->layers[net->num_layers - 1].p_tops[0];
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
    #ifdef DEMO
    const int out_layer_idx = net->num_layers - 1;
    #else
    const int out_layer_idx = net->num_layers - 2;
    #endif

    const Tensor* const out = net->layers[out_layer_idx].p_tops[0];
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
                    const unsigned char* const image_data,
                    const int height, const int width,
                    FILE* fp)
{
  set_input_pvanet(net, &image_data, &height, &width, 1);

  forward_net(net);

  get_output_pvanet(net, 0, fp);
}

void process_batch_pvanet(Net* const net,
                          const unsigned char* const * const images_data,
                          const int* const heights,
                          const int* const widths,
                          const int num_images,
                          FILE* fp)
{
  set_input_pvanet(net, images_data, heights, widths, num_images);

  forward_net(net);

  get_output_pvanet(net, 0, fp);
}
