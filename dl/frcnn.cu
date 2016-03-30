#include "layer.h"
#include <stdio.h>
#include <string.h>

static
void setup_frcnn_7_1_1(Net* const net)
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

    // Multi-scale RPN: 13 layers
    "rpn_conv1", "rpn_cls_score1", "rpn_bbox_pred1",
    "rpn_conv3", "rpn_cls_score3", "rpn_bbox_pred3",
    "rpn_conv5", "rpn_cls_score5", "rpn_bbox_pred5",
    "rpn_score", "rpn_pred", "rpn_bbox",
    "rpn_roi",

    // R-CNN: 10 layers
    "rcnn_roipool", "rcnn_roipool_flat",
    "fc6_L", "fc6_U", "fc7_L", "fc7_U",
    "cls_score", "cls_pred", "bbox_pred",
    "out", "test"
  };

  init_net(net);

  net->num_layers = 42;
  for (int i = 0; i < net->num_layers; ++i) {
    net->layers[i] = (Layer*)malloc(sizeof(Layer));
    init_layer(net->layers[i]);
    strcpy(net->layers[i]->name, names[i]);
  }

  net->img_info = (Tensor*)malloc(sizeof(Tensor));

  real anchor_scales[5] = { 3.0f, 6.0f, 9.0f, 16.0f, 32.0f };
  real anchor_ratios[5] = { 0.5f, 0.666f, 1.0f, 1.5f, 2.0f };
  memcpy(net->anchor_scales, anchor_scales, 5 * sizeof(real));
  memcpy(net->anchor_ratios, anchor_ratios, 5 * sizeof(real));

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

    for (int i = 17; i <= 26; ++i) {
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
      net->layers[21]->option.kernel_h = 3;
      net->layers[21]->option.kernel_w = 3;
      net->layers[21]->option.pad_h = 1;
      net->layers[21]->option.pad_w = 1;

      net->layers[24]->option.kernel_h = 5;
      net->layers[24]->option.kernel_w = 5;
      net->layers[24]->option.pad_h = 2;
      net->layers[24]->option.pad_w = 2;
    }

    {
      net->layers[16]->option.num_concats = 3;
      net->layers[27]->option.num_concats = 3;
      net->layers[29]->option.num_concats = 3;
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
    net->layers[18]->option.out_channels = 128;
    net->layers[19]->option.out_channels = 50;
    net->layers[20]->option.out_channels = 100;
    net->layers[21]->option.out_channels = 256;
    net->layers[22]->option.out_channels = 50;
    net->layers[23]->option.out_channels = 100;
    net->layers[24]->option.out_channels = 128;
    net->layers[25]->option.out_channels = 50;
    net->layers[26]->option.out_channels = 100;

    net->layers[30]->option.scales = &net->anchor_scales[0];
    net->layers[30]->option.ratios = &net->anchor_ratios[0];
    net->layers[30]->option.num_scales = 5;
    net->layers[30]->option.num_ratios = 5;
    net->layers[30]->option.num_concats = 3;
    net->layers[30]->option.base_size = 16;
    net->layers[30]->option.feat_stride = 16;
    net->layers[30]->option.min_size = 16;
    net->layers[30]->option.pre_nms_topn = 6000;
    net->layers[30]->option.post_nms_topn = 300;
    net->layers[30]->option.nms_thresh = 0.7f;

    net->layers[31]->option.pooled_height = 6;
    net->layers[31]->option.pooled_width = 6;
    net->layers[31]->option.spatial_scale = 0.0625;
    net->layers[31]->option.flatten = 1;

    for (int i = 33; i <= 39; ++i) {
      net->layers[i]->option.bias = 1;
      net->layers[i]->option.negative_slope = 0;
      net->layers[i]->option.threshold = 0.5f;
      net->layers[i]->option.test = 1;
      net->layers[i]->option.scaled = 0;
      #ifdef GPU
      net->layers[i]->option.handle = (void*)&net->cublas_handle;
      #endif
    }
    net->layers[33]->option.bias = 0;
    net->layers[35]->option.bias = 0;
    net->layers[33]->option.out_channels = 512;
    net->layers[34]->option.out_channels = 4096;
    net->layers[35]->option.out_channels = 128;
    net->layers[36]->option.out_channels = 4096;
    net->layers[37]->option.out_channels = 21;
    net->layers[39]->option.out_channels = 84;

    net->layers[40]->option.min_size = 16;
    net->layers[40]->option.score_thresh = 0.7f;
    net->layers[40]->option.nms_thresh = 0.3f;
  }

  {
    net->layers[0]->num_tops = 1;

    for (int i = 1; i <= 15; ++i) {
      net->layers[i]->num_bottoms = 1;
      net->layers[i]->num_tops = 1;
      net->layers[i]->num_params = 2;
    }
    net->layers[8]->num_params = 0;
    net->layers[15]->num_params = 1;

    net->layers[16]->num_bottoms = 3;
    net->layers[16]->num_tops = 1;

    for (int i = 17; i <= 26; ++i) {
      net->layers[i]->num_bottoms = 1;
      net->layers[i]->num_tops = 1;
      net->layers[i]->num_params = 2;
    }

    net->layers[27]->num_bottoms = 3;
    net->layers[27]->num_tops = 1;

    net->layers[29]->num_bottoms = 3;
    net->layers[29]->num_tops = 1;

    net->layers[30]->num_bottoms = 3;
    net->layers[30]->num_tops = 1;
    net->layers[30]->num_aux_data = 1;

    net->layers[31]->num_bottoms = 2;
    net->layers[31]->num_tops = 1;

    for (int i = 33; i <= 39; ++i) {
      net->layers[i]->num_bottoms = 1;
      net->layers[i]->num_tops = 1;
      net->layers[i]->num_params = 2;
    }
    net->layers[33]->num_params = 1;
    net->layers[35]->num_params = 1;

    net->layers[38]->num_bottoms = 2;
    net->layers[38]->num_params = 0;

    net->layers[39]->num_bottoms = 2;

    net->layers[40]->num_bottoms = 4;
    net->layers[40]->num_tops = 1;

    net->layers[41]->num_bottoms = 4;
    net->layers[41]->num_tops = 1;
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
    net->layers[19]->allocate_top_data[0] = 1;
    net->layers[20]->allocate_top_data[0] = 1;
    net->layers[22]->allocate_top_data[0] = 1;
    net->layers[23]->allocate_top_data[0] = 1;
    net->layers[25]->allocate_top_data[0] = 1;
    net->layers[26]->allocate_top_data[0] = 1;
    net->layers[30]->allocate_top_data[0] = 1;
  }
}

static
void connect_frcnn_7_1_1(Net* const net)
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

  // Multi-scale RPN
  {
    // rpn_1, 3, 5
    for (int i = 18; i <= 26; i += 3) {
      // rpn_conv1, 3, 5
      net->layers[i]->p_bottoms[0] = &net->layers[17]->tops[0];
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

    // rpn_score
    net->layers[27]->p_bottoms[0] = &net->layers[19]->tops[0];
    net->layers[27]->p_bottoms[1] = &net->layers[22]->tops[0];
    net->layers[27]->p_bottoms[2] = &net->layers[25]->tops[0];
    net->layers[27]->f_forward[0] = forward_concat_layer;
    net->layers[27]->f_forward[1] = forward_rpn_pred_layer;
    net->layers[27]->f_shape[0] = shape_concat_layer;
    net->layers[27]->f_shape[1] = shape_rpn_pred_layer;

    // rpn_bbox
    net->layers[29]->p_bottoms[0] = &net->layers[20]->tops[0];
    net->layers[29]->p_bottoms[1] = &net->layers[23]->tops[0];
    net->layers[29]->p_bottoms[2] = &net->layers[26]->tops[0];
    net->layers[29]->f_forward[0] = forward_concat_layer;
    net->layers[29]->f_forward[1] = forward_rpn_bbox_layer;
    net->layers[29]->f_shape[0] = shape_concat_layer;
    net->layers[29]->f_shape[1] = shape_rpn_bbox_layer;

    // proposal
    net->layers[30]->p_bottoms[0] = &net->layers[27]->tops[0];
    net->layers[30]->p_bottoms[1] = &net->layers[29]->tops[0];
    net->layers[30]->p_bottoms[2] = net->img_info;
    net->layers[30]->f_forward[0] = forward_proposal_layer;
    net->layers[30]->f_shape[0] = shape_proposal_layer;
    net->layers[30]->f_init[0] = init_proposal_layer;
  }

  // R-CNN
  {
    // roipool
    net->layers[31]->p_bottoms[0] = &net->layers[17]->tops[0];
    net->layers[31]->p_bottoms[1] = &net->layers[30]->tops[0];
    net->layers[31]->f_forward[0] = forward_roipool_layer;
    net->layers[31]->f_shape[0] = shape_roipool_layer;

    // fc6_L, 6_U, 7_L, 7_U
    for (int i = 33; i <= 36; i += 2) {
      net->layers[i]->p_bottoms[0] = &net->layers[i - 1]->tops[0];
      net->layers[i]->f_forward[0] = forward_fc_layer;
      net->layers[i]->f_shape[0] = shape_fc_layer;

      net->layers[i + 1]->p_bottoms[0] = &net->layers[i]->tops[0];
      net->layers[i + 1]->f_forward[0] = forward_fc_layer;
      net->layers[i + 1]->f_forward[1] = forward_inplace_relu_layer;
      net->layers[i + 1]->f_forward[2] = forward_inplace_dropout_layer;
      net->layers[i + 1]->f_shape[0] = shape_fc_layer;
    }
    net->layers[33]->p_bottoms[0] = &net->layers[31]->tops[0];

    // score
    net->layers[37]->p_bottoms[0] = &net->layers[36]->tops[0];
    net->layers[37]->f_forward[0] = forward_fc_layer;
    net->layers[37]->f_shape[0] = shape_fc_layer;

    // pred
    net->layers[38]->p_bottoms[0] = &net->layers[37]->tops[0];
    net->layers[38]->p_bottoms[1] = &net->layers[30]->tops[0];
    net->layers[38]->f_forward[0] = forward_rcnn_pred_layer;
    //net->layers[38]->f_forward[1] = save_layer_tops;
    net->layers[38]->f_shape[0] = shape_rcnn_pred_layer;

    // bbox
    net->layers[39]->p_bottoms[0] = &net->layers[36]->tops[0];
    net->layers[39]->p_bottoms[1] = &net->layers[30]->tops[0];
    net->layers[39]->f_forward[0] = forward_fc_layer;
    net->layers[39]->f_forward[1] = forward_rcnn_bbox_layer;
    //net->layers[39]->f_forward[2] = save_layer_tops;
    net->layers[39]->f_shape[0] = shape_fc_layer;
    net->layers[39]->f_shape[1] = shape_rcnn_bbox_layer;

    // out
    net->layers[40]->p_bottoms[0] = &net->layers[38]->tops[0];
    net->layers[40]->p_bottoms[1] = &net->layers[39]->tops[0];
    net->layers[40]->p_bottoms[2] = &net->layers[30]->tops[0];
    net->layers[40]->p_bottoms[3] = net->img_info;
    net->layers[40]->f_forward[0] = forward_odout_layer;
    net->layers[40]->f_shape[0] = shape_odout_layer;

    // test
    net->layers[41]->p_bottoms[0] = &net->layers[38]->tops[0];
    net->layers[41]->p_bottoms[1] = &net->layers[39]->tops[0];
    net->layers[41]->p_bottoms[2] = &net->layers[30]->tops[0];
    net->layers[41]->p_bottoms[3] = net->img_info;
    net->layers[41]->f_forward[0] = forward_odtest_layer;
    net->layers[41]->f_shape[0] = shape_odtest_layer;
  }
}

void construct_frcnn_7_1_1(Net* net)
{
  long int space_cpu = 0;

  setup_frcnn_7_1_1(net);

  connect_frcnn_7_1_1(net);

  shape_net(net);

  printf("Max layer size = %ld\n", net->layer_size);
  printf("Max param size = %ld\n", net->param_size);
  printf("Max temp size = %ld\n", net->temp_size);
  printf("Max tempint size = %ld\n", net->tempint_size);
  printf("Max const size = %ld\n", net->const_size);

  malloc_net(net);

  net->space_cpu += space_cpu;

  {
    for (int i = 0; i < net->num_layers; ++i) {
      for (int j = 0; j < net->layers[i]->num_tops; ++j) {
        if (!net->layers[i]->allocate_top_data[j]) {
          net->layers[i]->tops[j].data = net->layer_data[j];
        }
      }
    }

    net->layers[1]->tops[0].data = net->layer_data[1];
    net->layers[3]->tops[0].data = net->layer_data[1];
    net->layers[5]->tops[0].data = net->layer_data[1];
    net->layers[7]->tops[0].data = net->layer_data[1];
    net->layers[10]->tops[0].data = net->layer_data[1];
    net->layers[12]->tops[0].data = net->layer_data[1];
    net->layers[14]->tops[0].data = net->layer_data[1];

    net->layers[11]->tops[0].data = net->layer_data[2];
    net->layers[15]->tops[0].data = net->layer_data[3];
    net->layers[17]->tops[0].data = net->layer_data[1];
    net->layers[27]->tops[0].data = net->layer_data[0];
    net->layers[29]->tops[0].data = net->layer_data[2];
    net->layers[31]->tops[0].data = net->layer_data[2];
    net->layers[34]->tops[0].data = net->layer_data[1];
    net->layers[36]->tops[0].data = net->layer_data[1];
    net->layers[37]->tops[0].data = net->layer_data[0];
    net->layers[38]->tops[0].data = net->layers[37]->tops[0].data;
    net->layers[39]->tops[0].data = net->layer_data[2];
    net->layers[40]->tops[0].data = net->layer_data[1];
    net->layers[41]->tops[0].data = net->layer_data[3];
  }

  init_layers(net);

  // print total memory size required
  {
  #ifdef GPU
    printf("%ldMB of main memory allocated\n",
           DIV_THEN_CEIL(net->space_cpu,  1000000));
    printf("%ldMB of GPU memory allocated\n",
           DIV_THEN_CEIL(net->space,  1000000));
  #else
    printf("%ldMB of main memory allocated\n",
           DIV_THEN_CEIL(net->space_cpu + net->space,  1000000));
  #endif
  }
}

static
void prepare_input(Net* net,
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

static
void get_output(Net* net, const int image_start_index, FILE* fp)
{
  // retrieve & print output
  {
    const Tensor* const out = &net->layers[40]->tops[0];
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
    const Tensor* const out = &net->layers[41]->tops[0];
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
  Net frcnn;
  construct_frcnn_7_1_1(&frcnn);

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
        prepare_input(&frcnn, (const char * const *)&line, count);

        // forward-pass
        forward_net(&frcnn);

        // retrieve output & save to file
        get_output(&frcnn, total_count, fp_out);

        total_count += count;
        count = 0;
        buf_count = 0;
      }
    }

    if (count > 0) {
      prepare_input(&frcnn, (const char * const *)&line, count);
      forward_net(&frcnn);
      get_output(&frcnn, total_count, fp_out);
    }

    fclose(fp_list);
    fclose(fp_out);
  }


  // end
  free_net(&frcnn);

  return 0;
}
#endif
