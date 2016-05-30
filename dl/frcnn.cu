#include "layer.h"
#include <string.h>

//#define MSRPN
#define FC_COMPRESS
#define INCEPTION

#define DROPOUT_SCALE_TRAIN 1  // for new PVANET
//#define DROPOUT_SCALE_TRAIN 0  // for old PVANET

static
void setup_data_layer(Net* const net)
{
  net->layers[0] = (Layer*)malloc(sizeof(Layer));
  init_layer(net->layers[0]);
  strcpy(net->layers[0]->name, "data");

  net->layers[0]->num_tops = 1;

  net->space_cpu += malloc_layer(net->layers[0]);

  {
    Tensor* input = &net->layers[0]->tops[0];
    input->num_items = BATCH_SIZE;
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
                    const int* const strides,
                    const Layer* const prev_layer,
                    void* const blas_handle,
                    long int* const p_space_cpu)
{
  for (int i = 0; i < num_layers; ++i) {
    layers[i] = (Layer*)malloc(sizeof(Layer));
    init_layer(layers[i]);
    strcpy(layers[i]->name, names[i]);

    layers[i]->option.kernel_h = kernels[i];
    layers[i]->option.kernel_w = kernels[i];
    layers[i]->option.stride_h = strides[i];
    layers[i]->option.stride_w = strides[i];
    layers[i]->option.pad_h = kernels[i] / 2;
    layers[i]->option.pad_w = kernels[i] / 2;
    layers[i]->option.out_channels = out_channels[i];
    layers[i]->option.num_groups = 1;
    layers[i]->option.bias = 1;
    layers[i]->option.handle = blas_handle;
    layers[i]->option.negative_slope = 0;
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
                         void* const blas_handle,
                         long int* const p_space_cpu)
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
    layers[i]->option.bias = 1;
    layers[i]->option.handle = blas_handle;
    layers[i]->option.negative_slope = 0;
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
void setup_hyper_sub(Layer** const layers,
                     const int* const out_channels,
                     const Layer* const downsample_layer,
                     const Layer* const as_is_layer,
                     const Layer* const upsample_layer,
                     void* const blas_handle,
                     long int* const p_space_cpu)
{
  const char* names[] = {
    "downsample", "upsample", "concat", "convf"
  };
  const int num_layers = 4;

  for (int i = 0; i < num_layers; ++i) {
    layers[i] = (Layer*)malloc(sizeof(Layer));
    init_layer(layers[i]);
    strcpy(layers[i]->name, names[i]);
  }

  // downsample
  layers[0]->option.kernel_h = 3;
  layers[0]->option.kernel_w = 3;
  layers[0]->option.stride_h = 2;
  layers[0]->option.stride_w = 2;
  layers[0]->option.pad_h = 0;
  layers[0]->option.pad_w = 0;
  layers[0]->num_bottoms = 1;
  layers[0]->num_tops = 1;
  layers[0]->num_params = 0;

  // upsample
  layers[1]->option.kernel_h = 4;
  layers[1]->option.kernel_w = 4;
  layers[1]->option.stride_h = 2;
  layers[1]->option.stride_w = 2;
  layers[1]->option.pad_h = 1;
  layers[1]->option.pad_w = 1;
  layers[1]->option.out_channels = out_channels[0];
  layers[1]->option.num_groups = out_channels[0];
  layers[1]->option.bias = 0;
  layers[1]->option.handle = blas_handle;
  layers[1]->option.negative_slope = 0;
  layers[1]->num_bottoms = 1;
  layers[1]->num_tops = 1;
  layers[1]->num_params = 1;

  // concat
  layers[2]->option.num_concats = 3;
  layers[2]->num_bottoms = 3;
  layers[2]->num_tops = 1;
  layers[2]->num_params = 0;

  // convf
  layers[3]->option.kernel_h = 1;
  layers[3]->option.kernel_w = 1;
  layers[3]->option.stride_h = 1;
  layers[3]->option.stride_w = 1;
  layers[3]->option.pad_h = 0;
  layers[3]->option.pad_w = 0;
  layers[3]->option.out_channels = out_channels[1];
  layers[3]->option.num_groups = 1;
  layers[3]->option.bias = 1;
  layers[3]->option.handle = blas_handle;
  layers[3]->option.negative_slope = 0;
  layers[3]->num_bottoms = 1;
  layers[3]->num_tops = 1;
  layers[3]->num_params = 2;

  for (int i = 0; i < num_layers; ++i) {
    *p_space_cpu += malloc_layer(layers[i]);
  }

  // downsample
  layers[0]->p_bottoms[0] = &downsample_layer->tops[0];
  layers[0]->f_forward[0] = forward_pool_layer;
  layers[0]->f_shape[0] = shape_pool_layer;

  // upsample
  layers[1]->p_bottoms[0] = &upsample_layer->tops[0];
  layers[1]->f_forward[0] = forward_deconv_layer;
  layers[1]->f_shape[0] = shape_deconv_layer;

  // concat
  layers[2]->p_bottoms[0] = &layers[0]->tops[0];
  layers[2]->p_bottoms[1] = &as_is_layer->tops[0];
  layers[2]->p_bottoms[2] = &layers[1]->tops[0];
  layers[2]->f_forward[0] = forward_concat_layer;
  layers[2]->f_shape[0] = shape_concat_layer;

  // convf
  layers[3]->p_bottoms[0] = &layers[2]->tops[0];
  layers[3]->f_forward[0] = forward_conv_layer;
  layers[3]->f_forward[1] = forward_inplace_relu_layer;
  layers[3]->f_shape[0] = shape_conv_layer;
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
    if (!layers[i]->allocate_top_data[0]) {
      layers[i]->tops[0].data = net->layer_data[i % 2];
    }
  }

  // final conv layer's tops[0] = layer_data[4]
  if (!layers[num_layers - 1]->allocate_top_data[0]) {
    layers[num_layers - 1]->tops[0].data = net->layer_data[4];
  }
}

static
void assign_inception_sub_tops(Net* const net,
                               Layer* const * const layers,
                               const int stride)
{
  // assume prev_layer->tops[0] = layer_data[4]

  // pool1, conv1
  if (stride > 1) {
    layers[0]->tops[0].data = net->layer_data[0];
  }
  layers[1]->tops[0].data = net->layer_data[1];

  // conv3_1, conv3_2
  layers[2]->tops[0].data = net->layer_data[0];
  layers[3]->tops[0].data = net->layer_data[2];

  // conv5_1, conv5_2, conv5_3
  layers[4]->tops[0].data = net->layer_data[0];
  layers[5]->tops[0].data = net->layer_data[4];
  layers[6]->tops[0].data = net->layer_data[3];

  // concat
  if (!layers[7]->allocate_top_data[0]) {
    layers[7]->tops[0].data = net->layer_data[4];
  }
}

static
void assign_hyper_sub_tops(Net* const net,
                           Layer* const * const layers)
{
  // assume prev_layer->tops[0] = layer_data[4]

  layers[0]->tops[0].data = net->layer_data[0];
  layers[1]->tops[0].data = net->layer_data[1];
  layers[2]->tops[0].data = net->layer_data[2];
  layers[3]->tops[0].data = net->layer_data[4];
}

static
void assign_inception_tops(Net* const net)
{
  int num_layers = 0;

  // data
  assign_data_tops(net);
  num_layers = 1;

  // conv1, conv2, conv3
  {
    const int sub_size = 3;

    assign_conv_sub_tops(net, &net->layers[num_layers], sub_size);
    num_layers += sub_size;
  }

  // inc3
  {
    const int num_sub = 5;
    const int sub_size = 8;

    for (int i = 0; i < num_sub; ++i) {
      const int stride = (i == 0) ? 2 : 1;
      assign_inception_sub_tops(net, &net->layers[num_layers], stride);
      num_layers += sub_size;
    }
  }

  // inc4
  {
    const int num_sub = 5;
    const int sub_size = 8;

    for (int i = 0; i < num_sub; ++i) {
      const int stride = (i == 0) ? 2 : 1;
      assign_inception_sub_tops(net, &net->layers[num_layers], stride);
      num_layers += sub_size;
    }
  }

  // hypercolumn
  {
    const int sub_size = 4;

    assign_hyper_sub_tops(net, &net->layers[num_layers]);
    num_layers += sub_size;
  }
}

static
void setup_inception(Net* const net)
{
  int num_layers = 0;
  Layer* downsample_layer = NULL;
  Layer* as_is_layer = NULL;
  Layer* upsample_layer = NULL;

  // data
  setup_data_layer(net);
  num_layers = 1;

  // conv1, conv2, conv3
  {
    const int sub_size = 3;
    const char* conv_names[] = {
      "conv1", "conv2", "conv3"
    };
    const int out_channels[] = { 24, 48, 96 };
    const int kernels[] = { 7, 3, 3 };
    const int strides[] = { 2, 2, 2 };

    setup_conv_sub(
        &net->layers[num_layers],  conv_names,  sub_size,
        out_channels,  kernels,  strides,
        net->layers[num_layers - 1],
      #ifdef GPU
        (void*)&net->cublas_handle,
      #else
        NULL,
      #endif
        &net->space_cpu);
    num_layers += sub_size;
    downsample_layer = net->layers[num_layers - 1];
    downsample_layer->allocate_top_data[0] = 1;
  }

  // inc3
  {
    const int num_sub = 5;
    const int sub_size = 8;
    const char* sub_names[] = {
      "inc3a", "inc3b", "inc3c", "inc3d", "inc3e"
    };
    const int out_channels[] = { 96, 24, 64, 12, 24, 24 };

    for (int i = 0; i < num_sub; ++i) {
      const int stride = (i == 0) ? 2 : 1;
      setup_inception_sub(
          &net->layers[num_layers],  sub_names[i],
          out_channels,  stride,
          net->layers[num_layers - 1],
        #ifdef GPU
          (void*)&net->cublas_handle,
        #else
          NULL,
        #endif
          &net->space_cpu);
      num_layers += sub_size;
    }
    as_is_layer = net->layers[num_layers - 1];
    as_is_layer->allocate_top_data[0] = 1;
  }

  // inc4
  {
    const int num_sub = 5;
    const int sub_size = 8;
    const char* sub_names[] = {
      "inc4a", "inc4b", "inc4c", "inc4d", "inc4e"
    };
    const int out_channels[] = { 128, 32, 96, 16, 32, 32 };

    for (int i = 0; i < num_sub; ++i) {
      const int stride = (i == 0) ? 2 : 1;
      setup_inception_sub(
          &net->layers[num_layers],  sub_names[i],  out_channels,  stride,
          net->layers[num_layers - 1],
        #ifdef GPU
          (void*)&net->cublas_handle,
        #else
          NULL,
        #endif
          &net->space_cpu);
      num_layers += sub_size;
    }
    upsample_layer = net->layers[num_layers - 1];
  }

  // hypercolumn
  {
    const int sub_size = 4;
    const int out_channels[] = { 256, 256 };

    setup_hyper_sub(
        &net->layers[num_layers],  out_channels,
        downsample_layer,  as_is_layer,  upsample_layer,
        #ifdef GPU
          (void*)&net->cublas_handle,
        #else
          NULL,
        #endif
          &net->space_cpu);
    num_layers += sub_size;
  }

  net->num_layers = num_layers;
  net->num_layer_data = 5;
}

static
void setup_pva711(Net* const net)
{
  int num_layers = 0;
  Layer* downsample_layer = NULL;
  Layer* as_is_layer = NULL;
  Layer* upsample_layer = NULL;

  // data
  setup_data_layer(net);
  num_layers = 1;

  // conv1, conv2, conv3
  {
    const int sub_size = 13;
    const char* conv_names[] = {
      "conv1_1", "conv1_2",
      "conv2_1", "conv2_2",
      "conv3_1", "conv3_2", "conv3_3",
      "conv4_1", "conv4_2", "conv4_3",
      "conv5_1", "conv5_2", "conv5_3",
    };
    const int out_channels[] = {
      32, 32, 64, 64, 96, 64, 128, 192, 128, 256, 384, 256, 512
    };
    const int kernels[] = { 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 };
    const int strides[] = { 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1 };

    setup_conv_sub(
        &net->layers[num_layers],  conv_names,  sub_size,
        out_channels,  kernels,  strides,
        net->layers[num_layers - 1],
      #ifdef GPU
        (void*)&net->cublas_handle,
      #else
        NULL,
      #endif
        &net->space_cpu);

    downsample_layer = net->layers[num_layers + 6];
    downsample_layer->allocate_top_data[0] = 1;
    as_is_layer = net->layers[num_layers + 9];
    as_is_layer->allocate_top_data[0] = 1;
    upsample_layer = net->layers[num_layers + 12];

    num_layers += sub_size;
  }

  // hypercolumn
  {
    const int sub_size = 4;
    const int out_channels[] = { 512, 512 };

    setup_hyper_sub(
        &net->layers[num_layers],  out_channels,
        downsample_layer,  as_is_layer,  upsample_layer,
        #ifdef GPU
          (void*)&net->cublas_handle,
        #else
          NULL,
        #endif
          &net->space_cpu);
    num_layers += sub_size;
  }

  net->num_layers = num_layers;
  net->num_layer_data = 5;
}

static
void assign_pva711_tops(Net* const net)
{
  int num_layers = 0;

  // data
  assign_data_tops(net);
  num_layers = 1;

  // conv1, conv2, conv3
  {
    const int sub_size = 13;

    assign_conv_sub_tops(net, &net->layers[num_layers], sub_size);
    num_layers += sub_size;
  }

  // hypercolumn
  {
    const int sub_size = 4;

    assign_hyper_sub_tops(net, &net->layers[num_layers]);
    num_layers += sub_size;
  }
}

static
void setup_frcnn(Net* const net,
                 Layer** const layers,
                 const int rpn_channels,
                 const Layer* const convnet_out_layer)
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

  for (int i = 0; i < sub_size; ++i) {
    layers[i] = (Layer*)malloc(sizeof(Layer));
    init_layer(layers[i]);
    strcpy(layers[i]->name, names[i]);
  }

  {
    #ifdef MSRPN
    const int num_conv_layers = 9;
    #else
    const int num_conv_layers = 3;
    #endif

    for (int i = 0; i < num_conv_layers; ++i) {
      layers[i]->option.kernel_h = 1;
      layers[i]->option.kernel_w = 1;
      layers[i]->option.stride_h = 1;
      layers[i]->option.stride_w = 1;
      layers[i]->option.pad_h = 0;
      layers[i]->option.pad_w = 0;
      layers[i]->option.num_groups = 1;
      layers[i]->option.bias = 1;
      layers[i]->option.negative_slope = 0;
      #ifdef GPU
      layers[i]->option.handle = (void*)&net->cublas_handle;
      #endif

      layers[i]->num_bottoms = 1;
      layers[i]->num_tops = 1;
      layers[i]->num_params = 2;
    }
  }

  #ifdef MSRPN
  {
    // 1x1 RPN
    layers[0]->option.out_channels = rpn_channels / 4;
    layers[1]->option.out_channels = 18;
    layers[2]->option.out_channels = 36;

    // 3x3 RPN
    layers[3]->option.kernel_h = 3;
    layers[3]->option.kernel_w = 3;
    layers[3]->option.pad_h = 1;
    layers[3]->option.pad_w = 1;
    layers[3]->option.out_channels = rpn_channels / 2;
    layers[4]->option.out_channels = 18;
    layers[5]->option.out_channels = 36;

    // 5x5 RPN
    layers[6]->option.kernel_h = 5;
    layers[6]->option.kernel_w = 5;
    layers[6]->option.pad_h = 2;
    layers[6]->option.pad_w = 2;
    layers[6]->option.out_channels = rpn_channels / 4;
    layers[7]->option.out_channels = 18;
    layers[8]->option.out_channels = 36;

    // score concat
    layers[9]->option.num_concats = 3;
    layers[9]->num_bottoms = 3;
    layers[9]->num_tops = 1;

    // bbox concat
    layers[10]->option.num_concats = 3;
    layers[10]->num_bottoms = 3;
    layers[10]->num_tops = 1;
  }
  #else
  {
    // 3x3 RPN if using PVA-7.1.1
    #ifndef INCEPTION
    layers[0]->option.kernel_h = 3;
    layers[0]->option.kernel_w = 3;
    layers[0]->option.pad_h = 1;
    layers[0]->option.pad_w = 1;
    #endif

    layers[0]->option.out_channels = rpn_channels;
    layers[1]->option.out_channels = 50;
    layers[2]->option.out_channels = 100;
  }
  #endif

  // proposal, RoI-pooling
  {
  #ifdef MSRPN
    real anchor_scales[9] = { 3.0f, 6.0f, 9.0f,
                              4.0f, 8.0f, 16.0f,
                              7.0f, 13.0f, 32.0f };
    real anchor_ratios[3] = { 0.5f, 1.0f, 2.0f };
    layers[11]->option.num_scales = 9;
    layers[11]->option.num_ratios = 3;
  #else
    real anchor_scales[5] = { 3.0f, 6.0f, 9.0f, 16.0f, 32.0f };
    real anchor_ratios[5] = { 0.5f, 0.667f, 1.0f, 1.5f, 2.0f };
    layers[11]->option.num_scales = 5;
    layers[11]->option.num_ratios = 5;
  #endif

    memcpy(net->anchor_scales, anchor_scales,
           layers[11]->option.num_scales * sizeof(real));
    memcpy(net->anchor_ratios, anchor_ratios,
           layers[11]->option.num_ratios * sizeof(real));

    layers[11]->option.num_concats = 1;
    layers[11]->option.base_size = 16;
    layers[11]->option.feat_stride = 16;
    layers[11]->option.min_size = 16;
    layers[11]->option.pre_nms_topn = 6000;
    layers[11]->option.post_nms_topn = 300;
    layers[11]->option.nms_thresh = 0.7f;
    layers[11]->option.scales = &net->anchor_scales[0];
    layers[11]->option.ratios = &net->anchor_ratios[0];
    layers[11]->num_bottoms = 3;
    layers[11]->num_tops = 1;
    layers[11]->num_aux_data = 1;

    layers[12]->option.pooled_height = 6;
    layers[12]->option.pooled_width = 6;
    layers[12]->option.spatial_scale = 0.0625;
    layers[12]->option.flatten = 1;
    layers[12]->num_bottoms = 2;
    layers[12]->num_tops = 1;
  }

  // fc6, fc7, RCNN score, RCNN bbox
  for (int i = 13; i <= 19; ++i) {
    layers[i]->option.bias = 1;
    layers[i]->option.negative_slope = 0;
    layers[i]->option.threshold = 0.5f;
    layers[i]->option.test = 1;
    layers[i]->option.scaled = DROPOUT_SCALE_TRAIN;
    #ifdef GPU
    layers[i]->option.handle = (void*)&net->cublas_handle;
    #endif

    layers[i]->num_bottoms = 1;
    layers[i]->num_tops = 1;
    layers[i]->num_params = 2;
  }

  // fc6, fc7
  {
  #ifdef FC_COMPRESS
    layers[13]->option.out_channels = 512;
    layers[13]->option.bias = 0;
    layers[13]->num_params = 1;
    layers[15]->option.out_channels = 128;
    layers[15]->option.bias = 0;
    layers[15]->num_params = 1;
  #else
    layers[13]->num_bottoms = 0;
    layers[13]->num_tops = 0;
    layers[13]->num_params = 0;
    layers[15]->num_bottoms = 0;
    layers[15]->num_tops = 0;
    layers[15]->num_params = 0;
  #endif
    layers[14]->option.out_channels = 4096;
    layers[16]->option.out_channels = 4096;
  }

  // RCNN score
  layers[17]->option.out_channels = 22;

  // RCNN pred
  layers[18]->num_bottoms = 2;
  layers[18]->num_params = 0;

  // RCNN bbox
  layers[19]->option.out_channels = 88;
  layers[19]->num_bottoms = 2;

  // output
  layers[20]->option.min_size = 16;
  layers[20]->option.score_thresh = 0.7f;
  layers[20]->option.nms_thresh = 0.3f;
  layers[20]->num_bottoms = 4;
  layers[20]->num_tops = 1;

  // test
  #ifndef DEMO
  {
    layers[21]->num_bottoms = 4;
    layers[21]->num_tops = 1;
  }
  #endif

  for (int i = 0; i < sub_size; ++i) {
    net->space_cpu += malloc_layer(layers[i]);
  }

  // RPN
  {
    #ifdef MSRPN
    const int num_conv_layers = 9;
    #else
    const int num_conv_layers = 3;
    #endif

    for (int i = 0; i < num_conv_layers; i += 3) {
      // conv
      layers[i]->p_bottoms[0] = &convnet_out_layer->tops[0];
      layers[i]->f_forward[0] = forward_conv_layer;
      layers[i]->f_forward[1] = forward_inplace_relu_layer;
      layers[i]->f_shape[0] = shape_conv_layer;

      // score
      layers[i + 1]->p_bottoms[0] = &layers[i]->tops[0];
      layers[i + 1]->f_forward[0] = forward_conv_layer;
      layers[i + 1]->f_shape[0] = shape_conv_layer;

      // bbox
      layers[i + 2]->p_bottoms[0] = &layers[i]->tops[0];
      layers[i + 2]->f_forward[0] = forward_conv_layer;
      layers[i + 2]->f_shape[0] = shape_conv_layer;
    }
  }

  #ifdef MSRPN
  {
    // scores & bbox for multi-scale RPNs
    layers[1]->allocate_top_data[0] = 1;
    layers[2]->allocate_top_data[0] = 1;
    layers[4]->allocate_top_data[0] = 1;
    layers[5]->allocate_top_data[0] = 1;
    layers[7]->allocate_top_data[0] = 1;
    layers[8]->allocate_top_data[0] = 1;

    // score concat
    layers[9]->p_bottoms[0] = &layers[1]->tops[0];
    layers[9]->p_bottoms[1] = &layers[4]->tops[0];
    layers[9]->p_bottoms[2] = &layers[7]->tops[0];
    layers[9]->f_forward[0] = forward_concat_layer;
    layers[9]->f_forward[1] = forward_rpn_pred_layer;
    layers[9]->f_shape[0] = shape_concat_layer;
    layers[9]->f_shape[1] = shape_rpn_pred_layer;

    // bbox concat
    layers[10]->p_bottoms[0] = &layers[2]->tops[0];
    layers[10]->p_bottoms[1] = &layers[5]->tops[0];
    layers[10]->p_bottoms[2] = &layers[8]->tops[0];
    layers[10]->f_forward[0] = forward_concat_layer;
    layers[10]->f_forward[1] = forward_rpn_bbox_layer;
    layers[10]->f_shape[0] = shape_concat_layer;
    layers[10]->f_shape[1] = shape_rpn_bbox_layer;
  }
  #else
  {
    layers[1]->allocate_top_data[0] = 1;
    layers[1]->f_forward[1] = forward_rpn_pred_layer;
    layers[1]->f_shape[1] = shape_rpn_pred_layer;
    layers[2]->allocate_top_data[0] = 1;
    layers[2]->f_forward[1] = forward_rpn_bbox_layer;
    layers[2]->f_shape[1] = shape_rpn_bbox_layer;
  }
  #endif

  // proposal & RoI-pooling
  {
  #ifdef MSRPN
    layers[11]->p_bottoms[0] = &layers[9]->tops[0];
    layers[11]->p_bottoms[1] = &layers[10]->tops[0];
  #else
    layers[11]->p_bottoms[0] = &layers[1]->tops[0];
    layers[11]->p_bottoms[1] = &layers[2]->tops[0];
  #endif
    layers[11]->p_bottoms[2] = net->img_info;
    layers[11]->f_forward[0] = forward_proposal_layer;
    layers[11]->f_shape[0] = shape_proposal_layer;
    layers[11]->f_init[0] = init_proposal_layer;
    layers[11]->allocate_top_data[0] = 1;

    layers[12]->p_bottoms[0] = &convnet_out_layer->tops[0];
    layers[12]->p_bottoms[1] = &layers[11]->tops[0];
    layers[12]->f_forward[0] = forward_roipool_layer;
    layers[12]->f_shape[0] = shape_roipool_layer;
  }

  // fc6_L, 6_U, 7_L, 7_U
  for (int i = 13; i <= 16; i += 2) {
  #ifdef FC_COMPRESS
    layers[i]->p_bottoms[0] = &layers[i - 1]->tops[0];
    layers[i]->f_forward[0] = forward_fc_layer;
    layers[i]->f_shape[0] = shape_fc_layer;
    layers[i + 1]->p_bottoms[0] = &layers[i]->tops[0];
  #else
    layers[i + 1]->p_bottoms[0] = &layers[i - 1]->tops[0];
  #endif
    layers[i + 1]->f_forward[0] = forward_fc_layer;
    layers[i + 1]->f_forward[1] = forward_inplace_relu_layer;
    layers[i + 1]->f_forward[2] = forward_inplace_dropout_layer;
    layers[i + 1]->f_shape[0] = shape_fc_layer;
  }

  // RCNN score
  layers[17]->p_bottoms[0] = &layers[16]->tops[0];
  layers[17]->f_forward[0] = forward_fc_layer;
  layers[17]->f_shape[0] = shape_fc_layer;

  // RCNN pred
  layers[18]->p_bottoms[0] = &layers[17]->tops[0];
  layers[18]->p_bottoms[1] = &layers[11]->tops[0];
  layers[18]->f_forward[0] = forward_rcnn_pred_layer;
  layers[18]->f_shape[0] = shape_rcnn_pred_layer;

  // RCNN bbox
  layers[19]->p_bottoms[0] = &layers[16]->tops[0];
  layers[19]->p_bottoms[1] = &layers[11]->tops[0];
  layers[19]->f_forward[0] = forward_fc_layer;
  layers[19]->f_forward[1] = forward_rcnn_bbox_layer;
  layers[19]->f_shape[0] = shape_fc_layer;
  layers[19]->f_shape[1] = shape_rcnn_bbox_layer;

  // output
  layers[20]->p_bottoms[0] = &layers[18]->tops[0];
  layers[20]->p_bottoms[1] = &layers[19]->tops[0];
  layers[20]->p_bottoms[2] = &layers[11]->tops[0];
  layers[20]->p_bottoms[3] = net->img_info;
  layers[20]->f_forward[0] = forward_odout_layer;
  layers[20]->f_shape[0] = shape_odout_layer;

  // test
  #ifndef DEMO
  {
    layers[21]->p_bottoms[0] = &layers[18]->tops[0];
    layers[21]->p_bottoms[1] = &layers[19]->tops[0];
    layers[21]->p_bottoms[2] = &layers[11]->tops[0];
    layers[21]->p_bottoms[3] = net->img_info;
    layers[21]->f_forward[0] = forward_odtest_layer;
    layers[21]->f_shape[0] = shape_odtest_layer;
  }
  #endif

  net->num_layers += sub_size;
}

static
void assign_frcnn_tops(Net* const net)
{
  #ifdef DEMO
  const int sub_size = 21;
  #else
  const int sub_size = 22;
  #endif

  Layer** const layers = &net->layers[net->num_layers - sub_size];

  for (int i = 0; i < sub_size; ++i) {
    for (int j = 0; j < layers[i]->num_tops; ++j) {
      if (!layers[i]->allocate_top_data[j]) {
        layers[i]->tops[j].data = net->layer_data[j];
      }
    }
  }

  {
  #ifdef MSRPN
    layers[9]->tops[0].data = net->layer_data[0];
    layers[10]->tops[0].data = net->layer_data[2];
  #endif
    layers[12]->tops[0].data = net->layer_data[2];
  }

  {
  #ifdef FC_COMPRESS
    layers[14]->tops[0].data = net->layer_data[1];
    layers[16]->tops[0].data = net->layer_data[1];
  #else
    layers[14]->tops[0].data = net->layer_data[0];
    layers[16]->tops[0].data = net->layer_data[1];
  #endif
  }

  layers[17]->tops[0].data = net->layer_data[0];
  layers[18]->tops[0].data = layers[17]->tops[0].data;
  layers[19]->tops[0].data = net->layer_data[2];
  layers[20]->tops[0].data = net->layer_data[1];

  #ifndef DEMO
  {
    layers[21]->tops[0].data = net->layer_data[3];
  }
  #endif
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
              pvanet->layers[pvanet->num_layers - 1]->option.out_channels,
              pvanet->layers[pvanet->num_layers - 1]);

  shape_net(pvanet);

  printf("Max layer size = %ld\n", pvanet->layer_size);
  printf("Max param size = %ld\n", pvanet->param_size);
  printf("Max temp size = %ld\n", pvanet->temp_size);
  printf("Max tempint size = %ld\n", pvanet->tempint_size);
  printf("Max const size = %ld\n", pvanet->const_size);

  malloc_net(pvanet);

  #ifdef INCEPTION
  assign_inception_tops(pvanet);
  #else
  assign_pva711_tops(pvanet);
  #endif
  assign_frcnn_tops(pvanet);

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

  Tensor* const input = &net->layers[0]->tops[0];
  int shape_changed = (input->num_items != num_images);

  if (!shape_changed) {
    for (int n = 0; n < num_images; ++n) {
      if (net->img_info->data[n * 6 + 4] != (real)heights[n] ||
          net->img_info->data[n * 6 + 5] != (real)widths[n])
      {
        shape_changed = 1;
        break;
      }
    }
  }

  input->ndim = 3;
  input->num_items = 0;
  input->start[0] = 0;

  net->img_info->ndim = 1;
  net->img_info->num_items = 0;

  for (int n = 0; n < num_images; ++n) {
    img2input(images_data[n], input, net->img_info,
              (unsigned char*)net->temp_data,
              heights[n], widths[n]);
  }

  if (shape_changed) {
    printf("shape changed\n");
    print_tensor_info("data", input);
    print_tensor_info("img_info", net->img_info);
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
    const Tensor* const out = &net->layers[net->num_layers - 1]->tops[0];
    const long int output_size = flatten_size(out);

  #ifdef GPU
    cudaMemcpyAsync(net->output_cpu_data, out->data,
                    output_size * sizeof(real),
                    cudaMemcpyDeviceToHost);
  #else
    memcpy(net->output_cpu_data, out->data, output_size * sizeof(real));
  #endif

    if (fp) {
      for (int n = 0; n < out->num_items; ++n) {
        const real* const p_out_item = net->output_cpu_data + out->start[n];

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

    const Tensor* const out = &net->layers[out_layer_idx]->tops[0];
    const long int output_size = flatten_size(out);

  #ifdef GPU
    cudaMemcpyAsync(net->output_cpu_data, out->data,
                    output_size * sizeof(real),
                    cudaMemcpyDeviceToHost);
  #else
    memcpy(net->output_cpu_data, out->data, output_size * sizeof(real));
  #endif

    net->num_output_boxes = 0;
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
