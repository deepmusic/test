#include "layer.h"
#include <string.h>

#define MSRPN
#define FC_COMPRESS

static
void setup_data_layer(Net* const net)
{
  net->layers[0] = (Layer*)malloc(sizeof(Layer));
  init_layer(net->layers[0]);
  strcpy(net->layers[0]->name, "data");

  net->layers[0]->num_tops = 1;

  net->space_cpu += malloc_layer(layers[0]);

  {
    Tensor* input = &net->layers[0]->tops[0];
    input->num_items = 1;
    input->ndim = 3;
    for (int n = 0; n < input->num_items; ++n) {
      input->shape[n][0] = 3;
      input->shape[n][1] = 640;
      input->shape[n][2] = 1024;
      input->start[n] = n * 3 * 640 * 1024;
    }
  }

  net->img_info = (Tensor*)malloc(sizeof(Tensor));
}

static
void setup_conv_sub(Layer** const layers,
                    const char* const * const names,
                    const int num_layers,
                    const int* const out_channels,
                    const int* const kernels,
                    const int stride,
                    const Layer* const prev_layer,
                    const void* const blas_handle,
                    int* const p_space_cpu)
{
  for (int i = 0; i < num_layers; ++i) {
    layers[i] = (Layer*)malloc(sizeof(Layer));
    init_layer(layers[i]);
    strcpy(layers[i]->name, names[i]);

    layers[i]->option.kernel_h = kernels[i];
    layers[i]->option.kernel_w = kernels[i];
    layers[i]->option.stride_h = stride;
    layers[i]->option.stride_w = stride;
    layers[i]->option.pad_h = kernels[i] / 2;
    layers[i]->option.pad_w = kernels[i] / 2;
    layers[i]->option.out_channels = out_channels[i];
    layers[i]->option.num_groups = 1;
    layers[i]->option.negative_slope = 0;
    layers[i]->option.handle = blas_handle;
    layers[i]->num_bottoms = 1;
    layers[i]->num_tops = 1;
    layers[i]->num_params = 2;
  }

  for (int i = 0; i < num_layers; ++i) {
    *p_space_cpu += malloc_layer(layers[i]);
  }

  layers[0]->p_bottoms[0] = &prev_layer->tops[0];
  for (int i = 1; i < num_layers; ++i) {
    layers[i]->p_bottoms[0] = &layers[i - 1]->tops[0];
  }

  for (int i = 0; i < num_layers; ++i) {
    layers[i]->f_forward[0] = forward_conv_layer;
    layers[i]->f_forward[1] = forward_inplace_relu_layer;
    layers[i]->f_shape[0] = shape_conv_layer;
  }
}

static
void setup_inception_sub(Layer** const layers,
                         const char* const sub_name,
                         const int* const out_channels,
                         const int stride,
                         const Layer* const prev_layer,
                         const void* const blas_handle,
                         int* const p_space_cpu)
{
  const char* names[] = {
    "pool1", "conv1",
    "conv3_1", "conv3_2",
    "conv5_1", "conv5_2", "conv5_3",
    "concat"
  };
  const int num_layers = 8;

  for (int i = 0; i < num_layers; ++i) {
    layers[i] = (Layer*)malloc(sizeof(Layer));
    init_layer(layers[i]);
    sprintf(layers[i]->name, "%s_%s", sub_name, names[i]);
  }

  for (int i = 1; i < num_layers - 1; ++i) {
    layers[i]->option.kernel_h = 1;
    layers[i]->option.kernel_w = 1;
    layers[i]->option.stride_h = 1;
    layers[i]->option.stride_w = 1;
    layers[i]->option.pad_h = 0;
    layers[i]->option.pad_w = 0;
    layers[i]->option.out_channels = out_channels[i - 1];
    layers[i]->option.num_groups = 1;
    layers[i]->option.negative_slope = 0;
    layers[i]->option.handle = blas_handle;
    layers[i]->num_bottoms = 1;
    layers[i]->num_tops = 1;
    layers[i]->num_params = 2;
  }

  // pool1
  layers[0]->option.kernel_h = 3;
  layers[0]->option.kernel_w = 3;
  layers[0]->option.stride_h = stride;
  layers[0]->option.stride_w = stride;
  layers[0]->option.pad_h = 0;
  layers[0]->option.pad_w = 0;
  layers[0]->num_bottoms = (stride > 1) ? 1 : 0;
  layers[0]->num_tops = (stride > 1) ? 1 : 0;
  layers[0]->num_params = 0;

  // conv1

  // conv3_1

  // conv3_2
  layers[3]->option.kernel_h = 3;
  layers[3]->option.kernel_w = 3;
  layers[3]->option.stride_h = stride;
  layers[3]->option.stride_w = stride;
  layers[3]->option.pad_h = 1;
  layers[3]->option.pad_w = 1;

  // conv5_1

  // conv5_2
  layers[5]->option.kernel_h = 3;
  layers[5]->option.kernel_w = 3;
  layers[5]->option.pad_h = 1;
  layers[5]->option.pad_w = 1;

  // conv5_3
  layers[6]->option.kernel_h = 3;
  layers[6]->option.kernel_w = 3;
  layers[6]->option.stride_h = stride;
  layers[6]->option.stride_w = stride;
  layers[6]->option.pad_h = 1;
  layers[6]->option.pad_w = 1;

  // concat
  layers[7]->option.num_concats = 3;
  layers[7]->num_bottoms = 3;
  layers[7]->num_tops = 1;
  layers[7]->num_params = 0;

  for (int i = 0; i < num_layers; ++i) {
    *p_space_cpu += malloc_layer(layers[i]);
  }

  // pool1, conv1
  if (stride > 1) {
    layers[0]->p_bottoms[0] = &prev_layer->tops[0];
    layers[0]->f_forward[0] = forward_pool_layer;
    layers[0]->f_shape[0] = shape_pool_layer;
    layers[1]->p_bottoms[0] = &layers[0]->tops[0];
  }
  else {
    layers[1]->p_bottoms[0] = &prev_layer->tops[0];
  }

  // conv3_1, conv3_2
  layers[2]->p_bottoms[0] = &prev_layer->tops[0];
  layers[3]->p_bottoms[0] = &layers[2]->tops[0];

  // conv5_1, conv5_2, conv5_3
  layers[4]->p_bottoms[0] = &prev_layer->tops[0];
  layers[5]->p_bottoms[0] = &layers[4]->tops[0];
  layers[6]->p_bottoms[0] = &layers[5]->tops[0];

  // conv*
  for (int i = 1; i < num_layers - 1; ++i) {
    layers[i]->f_forward[0] = forward_conv_layer;
    layers[i]->f_forward[1] = forward_inplace_relu_layer;
    layers[i]->f_shape[0] = shape_conv_layer;
  }

  // concat
  layers[7]->p_bottoms[0] = &layers[1]->tops[0];
  layers[7]->p_bottoms[1] = &layers[3]->tops[0];
  layers[7]->p_bottoms[2] = &layers[6]->tops[0];
  layers[7]->f_forward[0] = forward_concat_layer;
  layers[7]->f_shape[0] = shape_concat_layer;
}

static
void assign_data_tops(Net* const net)
{
  net->layers[0]->tops[0].data = net->layer_data[4];
}

static
void assign_conv_sub_tops(Net* const net,
                          Layer* const * const layers,
                          const int num_layers)
{
  // assume prev_layer->tops[0] = layer_data[4]
  for (int i = 0; i < num_layers - 1; ++i) {
    layers[i]->tops[0].data = net->layer_data[i % 2];
  }

  // final conv layer's tops[0] = layer_data[4]
  layers[num_layers - 1]->tops[0].data = net->layer_data[4];
}

static
void assign_inception_sub_tops(Net* const net,
                               Layer* const * const layers)
{
  // assume prev_layer->tops[0] = layer_data[4]

  // pool1, conv1
  layers[0]->tops[0].data = net->layer_data[0];
  layers[1]->tops[0].data = net->layer_data[1];

  // conv3_1, conv3_2
  layers[2]->tops[0].data = net->layer_data[0];
  layers[3]->tops[0].data = net->layer_data[2];

  // conv5_1, conv5_2, conv5_3
  layers[4]->tops[0].data = net->layer_data[0];
  layers[5]->tops[0].data = net->layer_data[4];
  layers[6]->tops[0].data = net->layer_data[3];

  // concat
  layers[7]->tops[0].data = net->layer_data[4];
}

static
void assign_inception_tops(Net* const net)
{
  int num_layers = 0;

  // data
  assign_data_tops(net);
  num_layers = 1;

  // conv1, conv2, conv3
  assign_conv_sub_tops(net, &net->layers[num_layers], 3);
  num_layers += 3;

  // inc3
  for (int i = 0; i < 5; ++i) {
    assign_inception_sub_tops(net, &net->layers[num_layers]);
    num_layers += 8;
  }

  // inc4
  for (int i = 0; i < 5; ++i) {
    assign_inception_sub_tops(net, &net->layers[num_layers]);
    num_layers += 8;
  }
}

static
void setup_inception(Net* const net)
{
  int num_layers = 0;

  init_net(net);

  // data
  setup_data_layer(net);
  num_layers = 1;

  // conv1, conv2, conv3
  {
    const char* conv_names[] = {
      "conv1", "conv2", "conv3"
    };
    const int out_channels[] = { 24, 48, 96 };
    const int kernels[] = { 7, 3, 3 };

    setup_conv_sub(
        &net->layers[num_layers],  conv_names,  3,
        out_channels,  kernels,  stride,
        net->layers[num_layers - 1],  (void*)&net->cublas_handle,
        &net->space_cpu);
    num_layers += 3;
  }

  // inc3
  {
    const char* sub_names[] = {
      "inc3a", "inc3b", "inc3c", "inc3d", "inc3e"
    };
    const int out_channels[] = { 96, 24, 64, 12, 24, 24 };

    for (int i = 0; i < 5; ++i) {
      const int stride = (i == 0) ? 2 : 1;
      setup_inception_sub(
          &net->layers[num_layers],  sub_names[i],
          out_channels,  stride,
          net->layers[num_layers - 1],  (void*)&net->cublas_handle,
          &net->space_cpu);
      num_layers += 8;
    }
  }

  // inc4
  {
    const char* sub_names[] = {
      "inc4a", "inc4b", "inc4c", "inc4d", "inc4e"
    }
    const int out_channels[] = { 128, 32, 96, 16, 32, 32 };

    for (int i = 0; i < 5; ++i) {
      const int stride = (i == 0) ? 2 : 1;
      setup_inception_sub(
          &net->layers[num_layers],  sub_names[i],  out_channels,  stride,
          net->layers[num_layers - 1],  (void*)&net->cublas_handle,
          &net->space_cpu);
      num_layers += 8;
    }
  }

  net->num_layers = num_layers;
  net->num_layer_data = 5;
}

static
void setup_pva711(Net* const net)
{
  const char* names[] = {
    // PVANET: 18 layers
    "data",
    "conv1_1", "conv1_2",
    "conv2_1", "conv2_2",
    "conv3_1", "conv3_2", "conv3_3", "downsample",
    "conv4_1", "conv4_2", "conv4_3",
    "conv5_1", "conv5_2", "conv5_3", "upsample",
    "concat", "convf",
  };

  init_net(net);

  net->num_layers = 18;
  for (int i = 0; i < net->num_layers; ++i) {
    net->layers[i] = (Layer*)malloc(sizeof(Layer));
    init_layer(net->layers[i]);
    strcpy(net->layers[i]->name, names[i]);
  }

  net->img_info = (Tensor*)malloc(sizeof(Tensor));

  net->num_layer_data = 4;

  {
    for (int i = 1; i <= 15; ++i) {
      net->layers[i]->option.num_groups = 1;
      net->layers[i]->option.kernel_h = 3;
      net->layers[i]->option.kernel_w = 3;
      net->layers[i]->option.pad_h = 1;
      net->layers[i]->option.pad_w = 1;
      net->layers[i]->option.bias = 1;
      net->layers[i]->option.stride_h = 2;
      net->layers[i]->option.stride_w = 2;
      net->layers[i]->option.negative_slope = 0;
      #ifdef GPU
      net->layers[i]->option.handle = (void*)&net->cublas_handle;
      #endif
    }
    {
      net->layers[8]->option.pad_h = 0;
      net->layers[8]->option.pad_w = 0;
      net->layers[8]->option.stride_h = 2;
      net->layers[8]->option.stride_w = 2;

      net->layers[15]->option.num_groups = 512;
      net->layers[15]->option.kernel_h = 4;
      net->layers[15]->option.kernel_w = 4;
      net->layers[15]->option.pad_h = 1;
      net->layers[15]->option.pad_w = 1;
      net->layers[15]->option.bias = 0;
      net->layers[15]->option.stride_h = 2;
      net->layers[15]->option.stride_w = 2;

      net->layers[16]->option.num_concats = 3;

      net->layers[17]->option.num_groups = 1;
      net->layers[17]->option.kernel_h = 1;
      net->layers[17]->option.kernel_w = 1;
      net->layers[17]->option.pad_h = 0;
      net->layers[17]->option.pad_w = 0;
      net->layers[17]->option.bias = 1;
      net->layers[17]->option.stride_h = 1;
      net->layers[17]->option.stride_w = 1;
      net->layers[17]->option.negative_slope = 0;
      #ifdef GPU
      net->layers[17]->option.handle = (void*)&net->cublas_handle;
      #endif

      net->layers[2]->option.stride_h = 1;
      net->layers[2]->option.stride_w = 1;

      net->layers[4]->option.stride_h = 1;
      net->layers[4]->option.stride_w = 1;

      net->layers[6]->option.stride_h = 1;
      net->layers[6]->option.stride_w = 1;

      net->layers[7]->option.stride_h = 1;
      net->layers[7]->option.stride_w = 1;

      net->layers[10]->option.stride_h = 1;
      net->layers[10]->option.stride_w = 1;

      net->layers[11]->option.stride_h = 1;
      net->layers[11]->option.stride_w = 1;

      net->layers[13]->option.stride_h = 1;
      net->layers[13]->option.stride_w = 1;

      net->layers[14]->option.stride_h = 1;
      net->layers[14]->option.stride_w = 1;
    }

    net->layers[1]->option.out_channels = 32;
    net->layers[2]->option.out_channels = 32;
    net->layers[3]->option.out_channels = 64;
    net->layers[4]->option.out_channels = 64;
    net->layers[5]->option.out_channels = 96;
    net->layers[6]->option.out_channels = 64;
    net->layers[7]->option.out_channels = 128;
    net->layers[9]->option.out_channels = 192;
    net->layers[10]->option.out_channels = 128;
    net->layers[11]->option.out_channels = 256;
    net->layers[12]->option.out_channels = 384;
    net->layers[13]->option.out_channels = 256;
    net->layers[14]->option.out_channels = 512;
    net->layers[15]->option.out_channels = 512;
    net->layers[17]->option.out_channels = 512;
  }

  {
    net->layers[0]->num_tops = 1;

    for (int i = 1; i <= 17; ++i) {
      net->layers[i]->num_bottoms = 1;
      net->layers[i]->num_tops = 1;
      net->layers[i]->num_params = 2;
    }
    net->layers[8]->num_params = 0;
    net->layers[15]->num_params = 1;

    net->layers[16]->num_bottoms = 3;
    net->layers[16]->num_tops = 1;
    net->layers[16]->num_params = 0;
  }

  for (int i = 0; i < net->num_layers; ++i) {
    net->space_cpu += malloc_layer(net->layers[i]);
  }

  {
    Tensor* input = &net->layers[0]->tops[0];
    input->num_items = 1;
    input->ndim = 3;
    for (int n = 0; n < input->num_items; ++n) {
      input->shape[n][0] = 3;
      input->shape[n][1] = 640;
      input->shape[n][2] = 1024;
      input->start[n] = n * 3 * 640 * 1024;
    }
  }

  {
    net->layers[8]->allocate_top_data[0] = 1;
  }
}

static
void setup_frcnn(Net* const net)
{
  const char* names[] = {
    // Multi-scale RPN: 12 layers
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

    // R-CNN: 10 layers
    "rcnn_roipool",
  #ifdef FC_COMPRESS
    "fc6_L", "fc6_U", "fc7_L", "fc7_U",
  #else
    "null", "fc6", "null", "fc7",
  #endif
    "cls_score", "cls_pred", "bbox_pred",
    "out", "test"
  };

  init_net(net);

  net->num_layers = 22;
  for (int i = 0; i < net->num_layers; ++i) {
    net->layers[i] = (Layer*)malloc(sizeof(Layer));
    init_layer(net->layers[i]);
    strcpy(net->layers[i]->name, names[i]);
  }

  //net->img_info = (Tensor*)malloc(sizeof(Tensor));

#ifdef MSRPN
  real anchor_scales[9] = { 3.0f, 6.0f, 9.0f,
                            4.0f, 8.0f, 16.0f,
                            7.0f, 13.0f, 32.0f };
  real anchor_ratios[3] = { 0.5f, 1.0f, 2.0f };
  memcpy(net->anchor_scales, anchor_scales, 9 * sizeof(real));
  memcpy(net->anchor_ratios, anchor_ratios, 3 * sizeof(real));
#else
  real anchor_scales[5] = { 3.0f, 6.0f, 9.0f, 16.0f, 32.0f };
  real anchor_ratios[5] = { 0.5f, 0.667f, 1.0f, 1.5f, 2.0f };
  memcpy(net->anchor_scales, anchor_scales, 5 * sizeof(real));
  memcpy(net->anchor_ratios, anchor_ratios, 5 * sizeof(real));
#endif

  net->num_layer_data = 4;

  {
    for (int i = 0; i <= 8; ++i) {
      net->layers[i]->option.num_groups = 1;
      net->layers[i]->option.kernel_h = 1;
      net->layers[i]->option.kernel_w = 1;
      net->layers[i]->option.pad_h = 0;
      net->layers[i]->option.pad_w = 0;
      net->layers[i]->option.bias = 1;
      net->layers[i]->option.stride_h = 1;
      net->layers[i]->option.stride_w = 1;
      net->layers[i]->option.negative_slope = 0;
      #ifdef GPU
      net->layers[i]->option.handle = (void*)&net->cublas_handle;
      #endif
    }
    {
    #ifdef MSRPN
      net->layers[3]->option.kernel_h = 3;
      net->layers[3]->option.kernel_w = 3;
      net->layers[3]->option.pad_h = 1;
      net->layers[3]->option.pad_w = 1;

      net->layers[6]->option.kernel_h = 5;
      net->layers[6]->option.kernel_w = 5;
      net->layers[6]->option.pad_h = 2;
      net->layers[6]->option.pad_w = 2;
    #else
      net->layers[0]->option.kernel_h = 3;
      net->layers[0]->option.kernel_w = 3;
      net->layers[0]->option.pad_h = 1;
      net->layers[0]->option.pad_w = 1;
    #endif
    }

    {
    #ifdef MSRPN
      net->layers[9]->option.num_concats = 3;
      net->layers[10]->option.num_concats = 3;
    #endif
    }

  #ifdef MSRPN
    net->layers[0]->option.out_channels = 128;
    net->layers[1]->option.out_channels = 18;
    net->layers[2]->option.out_channels = 36;
    net->layers[3]->option.out_channels = 256;
    net->layers[4]->option.out_channels = 18;
    net->layers[5]->option.out_channels = 36;
    net->layers[6]->option.out_channels = 128;
    net->layers[7]->option.out_channels = 18;
    net->layers[8]->option.out_channels = 36;
  #else
    net->layers[0]->option.out_channels = 512;
    net->layers[1]->option.out_channels = 50;
    net->layers[2]->option.out_channels = 100;
  #endif

    net->layers[11]->option.num_concats = 1;
    net->layers[11]->option.base_size = 16;
    net->layers[11]->option.feat_stride = 16;
    net->layers[11]->option.min_size = 16;
    net->layers[11]->option.pre_nms_topn = 6000;
    net->layers[11]->option.post_nms_topn = 300;
    net->layers[11]->option.nms_thresh = 0.7f;
    net->layers[11]->option.scales = &net->anchor_scales[0];
    net->layers[11]->option.ratios = &net->anchor_ratios[0];
  #ifdef MSRPN
    net->layers[11]->option.num_scales = 9;
    net->layers[11]->option.num_ratios = 3;
  #else
    net->layers[11]->option.num_scales = 5;
    net->layers[11]->option.num_ratios = 5;
  #endif

    net->layers[12]->option.pooled_height = 6;
    net->layers[12]->option.pooled_width = 6;
    net->layers[12]->option.spatial_scale = 0.0625;
    net->layers[12]->option.flatten = 1;

    for (int i = 13; i <= 19; ++i) {
      net->layers[i]->option.bias = 1;
      net->layers[i]->option.negative_slope = 0;
      net->layers[i]->option.threshold = 0.5f;
      net->layers[i]->option.test = 1;
      net->layers[i]->option.scaled = 1;
      #ifdef GPU
      net->layers[i]->option.handle = (void*)&net->cublas_handle;
      #endif
    }
    net->layers[13]->option.bias = 0;
    net->layers[15]->option.bias = 0;
    net->layers[13]->option.out_channels = 512;
    net->layers[14]->option.out_channels = 4096;
    net->layers[15]->option.out_channels = 128;
    net->layers[16]->option.out_channels = 4096;
    net->layers[17]->option.out_channels = 21;
    net->layers[19]->option.out_channels = 84;

    net->layers[20]->option.min_size = 16;
    net->layers[20]->option.score_thresh = 0.7f;
    net->layers[20]->option.nms_thresh = 0.3f;
  }

  {
    for (int i = 0; i <= 2; ++i) {
      net->layers[i]->num_bottoms = 1;
      net->layers[i]->num_tops = 1;
      net->layers[i]->num_params = 2;
    }

  #ifdef MSRPN
    for (int i = 3; i <= 8; ++i) {
      net->layers[i]->num_bottoms = 1;
      net->layers[i]->num_tops = 1;
      net->layers[i]->num_params = 2;
    }
    net->layers[9]->num_bottoms = 3;
    net->layers[9]->num_tops = 1;
    net->layers[10]->num_bottoms = 3;
    net->layers[10]->num_tops = 1;
  #endif

    net->layers[11]->num_bottoms = 3;
    net->layers[11]->num_tops = 1;
    net->layers[11]->num_aux_data = 1;

    net->layers[12]->num_bottoms = 2;
    net->layers[12]->num_tops = 1;

    for (int i = 13; i <= 19; ++i) {
      net->layers[i]->num_bottoms = 1;
      net->layers[i]->num_tops = 1;
      net->layers[i]->num_params = 2;
    }
  #ifdef FC_COMPRESS
    net->layers[13]->num_params = 1;
    net->layers[15]->num_params = 1;
  #else
    net->layers[13]->num_bottoms = 0;
    net->layers[13]->num_tops = 0;
    net->layers[13]->num_params = 0;
    net->layers[15]->num_bottoms = 0;
    net->layers[15]->num_tops = 0;
    net->layers[15]->num_params = 0;
  #endif

    net->layers[18]->num_bottoms = 2;
    net->layers[18]->num_params = 0;

    net->layers[19]->num_bottoms = 2;

    net->layers[20]->num_bottoms = 4;
    net->layers[20]->num_tops = 1;

    net->layers[21]->num_bottoms = 4;
    net->layers[21]->num_tops = 1;
  }

  for (int i = 0; i < net->num_layers; ++i) {
    net->space_cpu += malloc_layer(net->layers[i]);
  }

  {
    net->layers[1]->allocate_top_data[0] = 1;
    net->layers[2]->allocate_top_data[0] = 1;
    net->layers[11]->allocate_top_data[0] = 1;
  #ifdef MSRPN
    net->layers[4]->allocate_top_data[0] = 1;
    net->layers[5]->allocate_top_data[0] = 1;
    net->layers[7]->allocate_top_data[0] = 1;
    net->layers[8]->allocate_top_data[0] = 1;
  #endif
  }
}

static
void connect_pva711(Net* const net)
{
  // PVANET
  {
    // 1_1, 1_2, 2_1, 2_2, 3_1, 3_2, 3_3
    for (int i = 1; i <= 7; ++i) {
      net->layers[i]->p_bottoms[0] = &net->layers[i - 1]->tops[0];
      net->layers[i]->f_forward[0] = forward_conv_layer;
      net->layers[i]->f_forward[1] = forward_inplace_relu_layer;
      net->layers[i]->f_shape[0] = shape_conv_layer;
    }

    // downsample
    net->layers[8]->p_bottoms[0] = &net->layers[7]->tops[0];
    net->layers[8]->f_forward[0] = forward_pool_layer;
    net->layers[8]->f_shape[0] = shape_pool_layer;

    // 4_1, 4_2, 4_3, 5_1, 5_2, 5_3
    for (int i = 9; i <= 14; ++i) {
      net->layers[i]->p_bottoms[0] = &net->layers[i - 1]->tops[0];
      net->layers[i]->f_forward[0] = forward_conv_layer;
      net->layers[i]->f_forward[1] = forward_inplace_relu_layer;
      net->layers[i]->f_shape[0] = shape_conv_layer;
    }
    net->layers[9]->p_bottoms[0] = &net->layers[7]->tops[0];

    // upsample
    net->layers[15]->p_bottoms[0] = &net->layers[14]->tops[0];
    net->layers[15]->f_forward[0] = forward_deconv_layer;
    net->layers[15]->f_shape[0] = shape_deconv_layer;

    // concat
    net->layers[16]->p_bottoms[0] = &net->layers[8]->tops[0];
    net->layers[16]->p_bottoms[1] = &net->layers[11]->tops[0];
    net->layers[16]->p_bottoms[2] = &net->layers[15]->tops[0];
    net->layers[16]->f_forward[0] = forward_concat_layer;
    net->layers[16]->f_shape[0] = shape_concat_layer;

    // convf
    net->layers[17]->p_bottoms[0] = &net->layers[16]->tops[0];
    net->layers[17]->f_forward[0] = forward_conv_layer;
    net->layers[17]->f_forward[1] = forward_inplace_relu_layer;
    net->layers[17]->f_shape[0] = shape_conv_layer;
  }
}

static
void connect_frcnn(Net* const net,
                   const Net* const convnet)
{
  // Multi-scale RPN
  {
    // rpn_1, 3, 5
  #ifdef MSRPN
    const int rpn_layer_end = 8;
  #else
    const int rpn_layer_end = 2;
  #endif
    for (int i = 0; i <= rpn_layer_end; i += 3) {
      // rpn_conv1, 3, 5
      net->layers[i]->p_bottoms[0]
          = &convnet->layers[convnet->num_layers - 1]->tops[0];
      net->layers[i]->f_forward[0] = forward_conv_layer;
      net->layers[i]->f_forward[1] = forward_inplace_relu_layer;
      net->layers[i]->f_shape[0] = shape_conv_layer;

      // rpn_cls_score1, 3, 5
      net->layers[i + 1]->p_bottoms[0] = &net->layers[i]->tops[0];
      net->layers[i + 1]->f_forward[0] = forward_conv_layer;
      net->layers[i + 1]->f_shape[0] = shape_conv_layer;

      // rpn_bbox_pred1, 3, 5
      net->layers[i + 2]->p_bottoms[0] = &net->layers[i]->tops[0];
      net->layers[i + 2]->f_forward[0] = forward_conv_layer;
      net->layers[i + 2]->f_shape[0] = shape_conv_layer;
    }

  #ifdef MSRPN
    // rpn_score
    net->layers[9]->p_bottoms[0] = &net->layers[1]->tops[0];
    net->layers[9]->p_bottoms[1] = &net->layers[4]->tops[0];
    net->layers[9]->p_bottoms[2] = &net->layers[7]->tops[0];
    net->layers[9]->f_forward[0] = forward_concat_layer;
    net->layers[9]->f_forward[1] = forward_rpn_pred_layer;
    net->layers[9]->f_shape[0] = shape_concat_layer;
    net->layers[9]->f_shape[1] = shape_rpn_pred_layer;

    // rpn_bbox
    net->layers[10]->p_bottoms[0] = &net->layers[2]->tops[0];
    net->layers[10]->p_bottoms[1] = &net->layers[5]->tops[0];
    net->layers[10]->p_bottoms[2] = &net->layers[8]->tops[0];
    net->layers[10]->f_forward[0] = forward_concat_layer;
    net->layers[10]->f_forward[1] = forward_rpn_bbox_layer;
    net->layers[10]->f_shape[0] = shape_concat_layer;
    net->layers[10]->f_shape[1] = shape_rpn_bbox_layer;
  #else
    net->layers[1]->f_forward[1] = forward_rpn_pred_layer;
    net->layers[1]->f_shape[1] = shape_rpn_pred_layer;
    net->layers[2]->f_forward[1] = forward_rpn_bbox_layer;
    net->layers[2]->f_shape[1] = shape_rpn_bbox_layer;
  #endif

    // proposal
  #ifdef MSRPN
    net->layers[11]->p_bottoms[0] = &net->layers[9]->tops[0];
    net->layers[11]->p_bottoms[1] = &net->layers[10]->tops[0];
  #else
    net->layers[11]->p_bottoms[0] = &net->layers[1]->tops[0];
    net->layers[11]->p_bottoms[1] = &net->layers[2]->tops[0];
  #endif
    net->layers[11]->p_bottoms[2] = convnet->img_info;
    net->layers[11]->f_forward[0] = forward_proposal_layer;
    net->layers[11]->f_shape[0] = shape_proposal_layer;
    net->layers[11]->f_init[0] = init_proposal_layer;
  }

  // R-CNN
  {
    // roipool
    net->layers[12]->p_bottoms[0]
        = &convnet->layers[convnet->num_layers - 1]->tops[0];
    net->layers[12]->p_bottoms[1] = &net->layers[11]->tops[0];
    net->layers[12]->f_forward[0] = forward_roipool_layer;
    net->layers[12]->f_shape[0] = shape_roipool_layer;

    // fc6_L, 6_U, 7_L, 7_U
    for (int i = 13; i <= 16; i += 2) {
    #ifdef FC_COMPRESS
      net->layers[i]->p_bottoms[0] = &net->layers[i - 1]->tops[0];
      net->layers[i]->f_forward[0] = forward_fc_layer;
      net->layers[i]->f_shape[0] = shape_fc_layer;
      net->layers[i + 1]->p_bottoms[0] = &net->layers[i]->tops[0];
    #else
      net->layers[i + 1]->p_bottoms[0] = &net->layers[i - 1]->tops[0];
    #endif
      net->layers[i + 1]->f_forward[0] = forward_fc_layer;
      net->layers[i + 1]->f_forward[1] = forward_inplace_relu_layer;
      net->layers[i + 1]->f_forward[2] = forward_inplace_dropout_layer;
      net->layers[i + 1]->f_shape[0] = shape_fc_layer;
    }

    // score
    net->layers[17]->p_bottoms[0] = &net->layers[16]->tops[0];
    net->layers[17]->f_forward[0] = forward_fc_layer;
    net->layers[17]->f_shape[0] = shape_fc_layer;

    // pred
    net->layers[18]->p_bottoms[0] = &net->layers[17]->tops[0];
    net->layers[18]->p_bottoms[1] = &net->layers[11]->tops[0];
    net->layers[18]->f_forward[0] = forward_rcnn_pred_layer;
    //net->layers[18]->f_forward[1] = save_layer_tops;
    net->layers[18]->f_shape[0] = shape_rcnn_pred_layer;

    // bbox
    net->layers[19]->p_bottoms[0] = &net->layers[16]->tops[0];
    net->layers[19]->p_bottoms[1] = &net->layers[11]->tops[0];
    net->layers[19]->f_forward[0] = forward_fc_layer;
    net->layers[19]->f_forward[1] = forward_rcnn_bbox_layer;
    //net->layers[19]->f_forward[2] = save_layer_tops;
    net->layers[19]->f_shape[0] = shape_fc_layer;
    net->layers[19]->f_shape[1] = shape_rcnn_bbox_layer;

    // out
    net->layers[20]->p_bottoms[0] = &net->layers[18]->tops[0];
    net->layers[20]->p_bottoms[1] = &net->layers[19]->tops[0];
    net->layers[20]->p_bottoms[2] = &net->layers[11]->tops[0];
    net->layers[20]->p_bottoms[3] = convnet->img_info;
    net->layers[20]->f_forward[0] = forward_odout_layer;
    net->layers[20]->f_shape[0] = shape_odout_layer;

    // test
    net->layers[21]->p_bottoms[0] = &net->layers[18]->tops[0];
    net->layers[21]->p_bottoms[1] = &net->layers[19]->tops[0];
    net->layers[21]->p_bottoms[2] = &net->layers[11]->tops[0];
    net->layers[21]->p_bottoms[3] = convnet->img_info;
    net->layers[21]->f_forward[0] = forward_odtest_layer;
    net->layers[21]->f_shape[0] = shape_odtest_layer;
  }
}

void construct_frcnn_7_1_1(Net* const convnet, Net* const frcnn)
{
  //setup_pva711(convnet);
  setup_inception(convnet);
  setup_frcnn(frcnn);

  //connect_pva711(convnet);
  connect_frcnn(frcnn, convnet);

  shape_net(convnet);
  shape_net(frcnn);

  printf("Max layer size = %ld + %ld = %ld\n",
         convnet->layer_size, frcnn->layer_size,
         convnet->layer_size + frcnn->layer_size);
  printf("Max param size = %ld + %ld = %ld\n",
         convnet->param_size, frcnn->param_size,
         convnet->param_size + frcnn->param_size);
  printf("Max temp size = %ld + %ld = %ld\n",
         convnet->temp_size, frcnn->temp_size,
         convnet->temp_size + frcnn->temp_size);
  printf("Max tempint size = %ld + %ld = %ld\n",
         convnet->tempint_size, frcnn->tempint_size,
         convnet->tempint_size + frcnn->tempint_size);
  printf("Max const size = %ld + %ld = %ld\n",
         convnet->const_size, frcnn->const_size,
         convnet->const_size + frcnn->const_size);

  malloc_net(convnet);
  malloc_net(frcnn);

  assign_inception_tops(net);
/*
  {
    for (int i = 0; i < convnet->num_layers; ++i) {
      for (int j = 0; j < convnet->layers[i]->num_tops; ++j) {
        if (!convnet->layers[i]->allocate_top_data[j]) {
          convnet->layers[i]->tops[j].data = convnet->layer_data[j];
        }
      }
    }

    convnet->layers[1]->tops[0].data = convnet->layer_data[1];
    convnet->layers[3]->tops[0].data = convnet->layer_data[1];
    convnet->layers[5]->tops[0].data = convnet->layer_data[1];
    convnet->layers[7]->tops[0].data = convnet->layer_data[1];
    convnet->layers[10]->tops[0].data = convnet->layer_data[1];
    convnet->layers[12]->tops[0].data = convnet->layer_data[1];
    convnet->layers[14]->tops[0].data = convnet->layer_data[1];

    convnet->layers[11]->tops[0].data = convnet->layer_data[2];
    convnet->layers[15]->tops[0].data = convnet->layer_data[3];
    convnet->layers[17]->tops[0].data = convnet->layer_data[1];
  }
*/

  {
    for (int i = 0; i < frcnn->num_layers; ++i) {
      for (int j = 0; j < frcnn->layers[i]->num_tops; ++j) {
        if (!frcnn->layers[i]->allocate_top_data[j]) {
          frcnn->layers[i]->tops[j].data = frcnn->layer_data[j];
        }
      }
    }

  #ifdef MSRPN
    frcnn->layers[9]->tops[0].data = frcnn->layer_data[0];
    frcnn->layers[10]->tops[0].data = frcnn->layer_data[2];
  #endif
    frcnn->layers[12]->tops[0].data = frcnn->layer_data[2];

  #ifdef FC_COMPRESS
    frcnn->layers[14]->tops[0].data = frcnn->layer_data[1];
    frcnn->layers[16]->tops[0].data = frcnn->layer_data[1];
  #else
    frcnn->layers[14]->tops[0].data = frcnn->layer_data[0];
    frcnn->layers[16]->tops[0].data = frcnn->layer_data[1];
  #endif

    frcnn->layers[17]->tops[0].data = frcnn->layer_data[0];
    frcnn->layers[18]->tops[0].data = frcnn->layers[17]->tops[0].data;
    frcnn->layers[19]->tops[0].data = frcnn->layer_data[2];
    frcnn->layers[20]->tops[0].data = frcnn->layer_data[1];
    frcnn->layers[21]->tops[0].data = frcnn->layer_data[3];
  }

  init_layers(convnet);
  init_layers(frcnn);

  // print total memory size required
  {
  #ifdef GPU
    printf("%ld + %ld = %ldMB of main memory allocated\n",
           DIV_THEN_CEIL(convnet->space_cpu,  1000000),
           DIV_THEN_CEIL(frcnn->space_cpu,  1000000),
           DIV_THEN_CEIL(convnet->space_cpu + frcnn->space_cpu,  1000000));
    printf("%ld + %ld = %ldMB of GPU memory allocated\n",
           DIV_THEN_CEIL(convnet->space,  1000000),
           DIV_THEN_CEIL(frcnn->space,  1000000),
           DIV_THEN_CEIL(convnet->space + frcnn->space,  1000000));
  #else
    printf("%ld + %ld = %ldMB of main memory allocated\n",
           DIV_THEN_CEIL(convnet->space + convnet->space_cpu,  1000000),
           DIV_THEN_CEIL(frcnn->space + frcnn->space_cpu,  1000000),
           DIV_THEN_CEIL(convnet->space + convnet->space_cpu
                         + frcnn->space + frcnn->space_cpu,  1000000));
  #endif
  }
}

void get_input_frcnn_7_1_1(Net* const net,
                           const char* const filename[],
                           const int num_images)
{
  Tensor* input = &net->layers[0]->tops[0];
  //input->data = net->input_cpu_data;
  input->ndim = 3;
  input->num_items = 0;
  input->start[0] = 0;

  net->img_info->ndim = 1;
  net->img_info->num_items = 0;

  for (int i = 0; i < num_images; ++i) {
    load_image(filename[i], input, net->img_info, net->temp_data);
  }
/*
  #ifdef GPU
  cudaMemcpyAsync(net->layer_data[0], input->data,
                  flatten_size(input) * sizeof(real),
                  cudaMemcpyHostToDevice);
  #else
  memcpy(net->layer_data[0], input->data,
         flatten_size(input) * sizeof(real));
  #endif
  input->data = net->layer_data[0];
*/
  // network reshape
  shape_net(net);

  print_tensor_info("data", input);
  print_tensor_info("img_info", net->img_info);
}

void get_output_frcnn_7_1_1(Net* const net,
                            const int image_start_index,
                            FILE* fp)
{
  // retrieve & print output
  {
    const Tensor* const out = &net->layers[20]->tops[0];
    const long int output_size = flatten_size(out);

  #ifdef GPU
    cudaMemcpyAsync(net->output_cpu_data, out->data,
                    output_size * sizeof(real),
                    cudaMemcpyDeviceToHost);
  #else
    memcpy(net->output_cpu_data, out->data, output_size * sizeof(real));
  #endif

    for (int n = 0; n < out->num_items; ++n) {
      const int image_index = image_start_index + n;
      const real* const p_out_item = net->output_cpu_data + out->start[n];

      for (int i = 0; i < out->shape[n][0]; ++i) {
        const int class_index = (int)p_out_item[i * 6 + 0];

        printf("Image %d / Box %d: ", image_index, i);
        printf("class %d, score %f, p1 = (%.2f, %.2f), p2 = (%.2f, %.2f)\n",
               class_index, p_out_item[i * 6 + 5],
               p_out_item[i * 6 + 1], p_out_item[i * 6 + 2],
               p_out_item[i * 6 + 3], p_out_item[i * 6 + 4]);
      }
    }
  }

  // retrieve & save test output for measuring performance
  {
    const Tensor* const out = &net->layers[21]->tops[0];
    const long int output_size = flatten_size(out);

  #ifdef GPU
    cudaMemcpyAsync(net->output_cpu_data, out->data,
                    output_size * sizeof(real),
                    cudaMemcpyDeviceToHost);
  #else
    memcpy(net->output_cpu_data, out->data, output_size * sizeof(real));
  #endif

    for (int n = 0; n < out->num_items; ++n) {
      const real* const p_out_item = net->output_cpu_data + out->start[n];

      fwrite(&out->ndim, sizeof(int), 1, fp);
      fwrite(out->shape[n], sizeof(int), out->ndim, fp);
      fwrite(p_out_item, sizeof(real), out->shape[n][0] * 6, fp);
    }
  }
}

#ifdef TEST
int main(int argc, char* argv[])
{
  // CUDA initialization
  #ifdef GPU
  {
    printf("set device\n");
    cudaSetDevice(0);
  }
  #endif

  // PVANET construction
  Net convnet, frcnn;
  construct_frcnn_7_1_1(&convnet, &frcnn);

  // load a text file containing image filenames to be tested
  {
    char buf[10240];
    char* line[20];
    int total_count = 0, count = 0, buf_count = 0;
    FILE* fp_list = fopen(argv[1], "r");
    FILE* fp_out = fopen(argv[2], "wb");

    if (!fp_list) {
      printf("File not found: %s\n", argv[1]);
    }
    if (!fp_out) {
      printf("File write error: %s\n", argv[2]);
    }

    while (fgets(&buf[buf_count], 1024, fp_list)) {
      const Tensor* const input = &frcnn.layers[0]->tops[0];
      const int len = strlen(&buf[buf_count]);
      buf[buf_count + len - 1] = 0;
      line[count] = &buf[buf_count];
      ++count;
      buf_count += len;
      if (count == input->num_items) {
        // input data loading
        get_input_frcnn_7_1_1(&convnet, (const char * const *)&line, count);

        // forward-pass
        forward_net(&convnet);
        forward_net(&frcnn);

        // retrieve output & save to file
        get_output_frcnn_7_1_1(&frcnn, total_count, fp_out);

        total_count += count;
        count = 0;
        buf_count = 0;
      }
    }

    if (count > 0) {
      get_input_frcnn_7_1_1(&convnet, (const char * const *)&line, count);
      forward_net(&convnet);
      forward_net(&frcnn);
      get_output_frcnn_7_1_1(&frcnn, total_count, fp_out);
    }

    fclose(fp_list);
    fclose(fp_out);
  }


  // end
  free_net(&convnet);
  free_net(&frcnn);

  return 0;
}
#endif
