#include "layer.h"
#include <stdio.h>
#include <string.h>

void forward_frcnn_7_1_1(Net* const net)
{
  // PVANET
  {
    // 1_1, 1_2, 2_1, 2_2, 3_1, 3_2, 3_3
    for (int i = 1; i <= 7; ++i) {
      forward_conv_layer(net, net->layers[i]);
      forward_inplace_relu_layer(net, net->layers[i]);
    }

    // downsample
    forward_pool_layer(net, net->layers[8]);

    // 4_1, 4_2, 4_3, 5_1, 5_2, 5_3
    for (int i = 9; i <= 14; ++i) {
      forward_conv_layer(net, net->layers[i]);
      forward_inplace_relu_layer(net, net->layers[i]);
    }

    // upsample
    forward_deconv_layer(net, net->layers[15]);

    // concat
    forward_concat_layer(net, net->layers[16]);

    // convf
    forward_conv_layer(net, net->layers[17]);
    forward_inplace_relu_layer(net, net->layers[17]);
  }

  // Multi-scale RPN
  {
    // rpn_1, 3, 5
    for (int i = 18; i <= 26; i += 3) {
      // rpn_conv1, 3, 5
      forward_conv_layer(net, net->layers[i]);
      forward_inplace_relu_layer(net, net->layers[i]);

      // rpn_cls_score1, 3, 5
      forward_conv_layer(net, net->layers[i + 1]);

      // rpn_bbox_pred1, 3, 5
      forward_conv_layer(net, net->layers[i + 2]);
    }

    // rpn_score
    forward_concat_layer(net, net->layers[27]);

    // pred
    {
      Tensor* const pred = &net->layers[28]->tops[0];
      const Tensor* const score = net->layers[28]->p_bottoms[0];
      pred->ndim = 3;
      pred->num_items = score->num_items;
      for (int n = 0; n < score->num_items; ++n) {
        pred->shape[n][0] = 2;
        pred->shape[n][1]
            = score->shape[n][0] / 2 * score->shape[n][1];
        pred->shape[n][2] = score->shape[n][2];
        pred->start[n] = score->start[n];
      }
      softmax_inplace_forward(pred, net->temp_data);
      print_tensor_info(net->layers[28]->name, pred);
    }

    // pred reshape
    {
      Tensor* const pred = &net->layers[28]->tops[0];
      const Tensor* const score = net->layers[28]->p_bottoms[0];
      pred->ndim = 4;
      pred->num_items = score->num_items;
      for (int n = 0; n < score->num_items; ++n) {
        pred->shape[n][0] = 2;
        pred->shape[n][1] = score->shape[n][0] / 2;
        pred->shape[n][2] = score->shape[n][1];
        pred->shape[n][3] = score->shape[n][2];
      }
      print_tensor_info("rpn_pred_reshape", pred);
    }

    // rpn_bbox
    forward_concat_layer(net, net->layers[29]);

    // bbox reshape
    {
      Tensor* const bbox = &net->layers[29]->tops[0];
      bbox->ndim = 4;
      for (int n = 0; n < bbox->num_items; ++n) {
        const int C = bbox->shape[n][0];
        const int H = bbox->shape[n][1];
        const int W = bbox->shape[n][2];
        bbox->shape[n][0] = C / 4;
        bbox->shape[n][1] = 4;
        bbox->shape[n][2] = H;
        bbox->shape[n][3] = W;
      }
      print_tensor_info("rpn_bbox_reshape", bbox);
    }

    // proposal
    forward_proposal_layer(net, net->layers[30]);
  }

  // R-CNN
  {
    // roipool
    forward_roipool_layer(net, net->layers[31]);

    // roipool_flat
    {
      Tensor* const roipool_flat = &net->layers[32]->tops[0];
      const Tensor* const roipool = &net->layers[31]->tops[0];
      // calculate total number of RoI-pooled data
      int total_num_rois = 0;
      for (int n = 0; n < roipool->num_items; ++n) {
        total_num_rois += roipool->shape[n][0];
      }

      // reshape to 2d tensor: total_num_rois x (C * H * W)
      roipool_flat->ndim = 2;
      roipool_flat->num_items = 1;
      roipool_flat->shape[0][0] = total_num_rois;
      roipool_flat->shape[0][1] = roipool->shape[0][1]
                                  * roipool->shape[0][2]
                                  * roipool->shape[0][3];
      roipool_flat->start[0] = 0;
      print_tensor_info("rcnn_roipool_flat", roipool_flat);
    }

    // fc6_L, 6_U, 7_L, 7_U
    for (int i = 33; i <= 36; i += 2) {
      forward_fc_layer(net, net->layers[i]);

      forward_fc_layer(net, net->layers[i + 1]);
      forward_inplace_relu_layer(net, net->layers[i + 1]);
      forward_inplace_dropout_layer(net, net->layers[i + 1]);
    }

    // score
    forward_fc_layer(net, net->layers[37]);

    // pred
    {
      Tensor* const pred = &net->layers[38]->tops[0];
      const Tensor* const roipool_flat = &net->layers[32]->tops[0];
      pred->ndim = 4;
      pred->num_items = 1;
      pred->shape[0][0] = roipool_flat->shape[0][0];
      pred->shape[0][1] = net->layers[37]->option.out_channels;
      pred->shape[0][2] = 1;
      pred->shape[0][3] = 1;
      pred->start[0] = 0;
      softmax_inplace_forward(pred, net->temp_data);
      print_tensor_info("pred", pred);
    }

    // pred reshape
    {
      Tensor* const pred = &net->layers[38]->tops[0];
      const Tensor* const roipool = &net->layers[31]->tops[0];
      pred->ndim = 2;
      pred->num_items = roipool->num_items;
      {
        int total_size = 0;
        for (int n = 0; n < roipool->num_items; ++n) {
          pred->shape[n][0] = roipool->shape[n][0];
          pred->shape[n][1] = net->layers[37]->option.out_channels;
          pred->start[n] = total_size;
          total_size += pred->shape[n][0] * pred->shape[n][1];
        }
      }
      print_tensor_info("pred_reshape", pred);
    }

    // bbox
    forward_fc_layer(net, net->layers[39]);

    // bbox reshape
    {
      Tensor* const bbox = &net->layers[39]->tops[0];
      const Tensor* const roipool = &net->layers[31]->tops[0];
      bbox->ndim = 3;
      bbox->num_items = roipool->num_items;
      {
        int total_size = 0;
        for (int n = 0; n < roipool->num_items; ++n) {
          bbox->shape[n][0] = roipool->shape[n][0];
          bbox->shape[n][1] = net->layers[39]->option.out_channels / 4;
          bbox->shape[n][2] = 4;
          bbox->start[n] = total_size;
          total_size += bbox->shape[n][0] * bbox->shape[n][1]
                        * bbox->shape[n][2];
        }
      }
    }

    // out
    forward_odout_layer(net, net->layers[40]);
  }
}

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
    "out"
  };

  net->initialized = 0;

  net->num_layers = 41;
  for (int i = 0; i < net->num_layers; ++i) {
    net->layers[i] = (Layer*)malloc(sizeof(Layer));
    strcpy(net->layers[i]->name, names[i]);
  }
  net->img_info = (Tensor*)malloc(sizeof(Tensor));

  real anchor_scales[5] = { 3.0f, 6.0f, 9.0f, 16.0f, 32.0f };
  real anchor_ratios[5] = { 0.5f, 0.666f, 1.0f, 1.5f, 2.0f };
  memcpy(net->anchor_scales, anchor_scales, 5 * sizeof(real));
  memcpy(net->anchor_ratios, anchor_ratios, 5 * sizeof(real));

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
      memcpy(&net->layers[4]->option, &net->layers[2]->option,
             sizeof(LayerOption));
      memcpy(&net->layers[6]->option, &net->layers[2]->option,
             sizeof(LayerOption));
      memcpy(&net->layers[7]->option, &net->layers[2]->option,
             sizeof(LayerOption));
      memcpy(&net->layers[10]->option, &net->layers[2]->option,
             sizeof(LayerOption));
      memcpy(&net->layers[11]->option, &net->layers[2]->option,
             sizeof(LayerOption));
      memcpy(&net->layers[13]->option, &net->layers[2]->option,
             sizeof(LayerOption));
      memcpy(&net->layers[14]->option, &net->layers[2]->option,
             sizeof(LayerOption));
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
    for (int i = 0; i < net->num_layers; ++i) {
      net->layers[i]->num_bottoms = 0;
      net->layers[i]->num_tops = 0;
      net->layers[i]->num_params = 0;
    }

    net->layers[0]->num_bottoms = 0;
    net->layers[0]->num_tops = 1;
    net->layers[0]->num_params = 0;

    for (int i = 1; i <= 15; ++i) {
      net->layers[i]->num_bottoms = 1;
      net->layers[i]->num_tops = 1;
      net->layers[i]->num_params = 2;
    }
    net->layers[8]->num_params = 0;
    net->layers[15]->num_params = 1;

    net->layers[16]->num_bottoms = 3;
    net->layers[16]->num_tops = 1;
    net->layers[16]->num_params = 0;

    for (int i = 17; i <= 26; ++i) {
      net->layers[i]->num_bottoms = 1;
      net->layers[i]->num_tops = 1;
      net->layers[i]->num_params = 2;
    }

    net->layers[27]->num_bottoms = 3;
    net->layers[27]->num_tops = 1;
    net->layers[27]->num_params = 0;

    net->layers[28]->num_bottoms = 1;
    net->layers[28]->num_tops = 1;
    net->layers[28]->num_params = 0;

    net->layers[29]->num_bottoms = 3;
    net->layers[29]->num_tops = 1;
    net->layers[29]->num_params = 0;

    net->layers[30]->num_bottoms = 3;
    net->layers[30]->num_tops = 1;
    net->layers[30]->num_params = 0;

    net->layers[31]->num_bottoms = 2;
    net->layers[31]->num_tops = 1;
    net->layers[31]->num_params = 0;

    net->layers[32]->num_bottoms = 1;
    net->layers[32]->num_tops = 1;
    net->layers[32]->num_params = 0;

    for (int i = 33; i <= 39; ++i) {
      net->layers[i]->num_bottoms = 1;
      net->layers[i]->num_tops = 1;
      net->layers[i]->num_params = 2;
    }
    net->layers[33]->num_params = 1;
    net->layers[35]->num_params = 1;
    net->layers[38]->num_params = 0;

    net->layers[40]->num_bottoms = 4;
    net->layers[40]->num_tops = 1;
    net->layers[40]->num_params = 0;
  }

  {
    net->num_layers = 41;
    net->num_layer_data = 4;
    net->layer_size = 0;
    net->param_size = 0;
    net->temp_size = 0;
    net->tempint_size = 0;
    net->const_size = 0;
  }
}

void connect_frcnn_7_1_1(Net* const net)
{
  {
    // PVANET conv1_1, ..., conv5_3
    for (int i = 1; i <= 15; ++i) {
      net->layers[i]->p_bottoms[0] = &net->layers[i - 1]->tops[0];
    }
    net->layers[9]->p_bottoms[0] = &net->layers[7]->tops[0];
    // concat
    net->layers[16]->p_bottoms[0] = &net->layers[8]->tops[0];
    net->layers[16]->p_bottoms[1] = &net->layers[11]->tops[0];
    net->layers[16]->p_bottoms[2] = &net->layers[15]->tops[0];
    // convf
    net->layers[17]->p_bottoms[0] = &net->layers[16]->tops[0];
  }

  {
    // rpn_1, 3, 5
    for (int i = 18; i <= 26; i += 3) {
      net->layers[i]->p_bottoms[0] = &net->layers[17]->tops[0];
      net->layers[i + 1]->p_bottoms[0] = &net->layers[i]->tops[0];
      net->layers[i + 2]->p_bottoms[0] = &net->layers[i]->tops[0];
    }
    // rpn_score
    net->layers[27]->p_bottoms[0] = &net->layers[19]->tops[0];
    net->layers[27]->p_bottoms[1] = &net->layers[22]->tops[0];
    net->layers[27]->p_bottoms[2] = &net->layers[25]->tops[0];
    // rpn_pred
    net->layers[28]->p_bottoms[0] = &net->layers[27]->tops[0];
    // rpn_bbox
    net->layers[29]->p_bottoms[0] = &net->layers[20]->tops[0];
    net->layers[29]->p_bottoms[1] = &net->layers[23]->tops[0];
    net->layers[29]->p_bottoms[2] = &net->layers[26]->tops[0];
    // proposal
    net->layers[30]->p_bottoms[0] = &net->layers[28]->tops[0];
    net->layers[30]->p_bottoms[1] = &net->layers[29]->tops[0];
    net->layers[30]->p_bottoms[2] = net->img_info;
  }

  {
    // roipool
    net->layers[31]->p_bottoms[0] = &net->layers[17]->tops[0];
    net->layers[31]->p_bottoms[1] = &net->layers[30]->tops[0];
    // roipool_flat
    net->layers[32]->p_bottoms[0] = &net->layers[31]->tops[0];
    // fc6_L, 6_U, 7_L, 7_U
    for (int i = 33; i <= 36; ++i) {
      net->layers[i]->p_bottoms[0] = &net->layers[i - 1]->tops[0];
    }
    // score
    net->layers[37]->p_bottoms[0] = &net->layers[36]->tops[0];
    // pred
    net->layers[38]->p_bottoms[0] = &net->layers[37]->tops[0];
    // bbox
    net->layers[39]->p_bottoms[0] = &net->layers[36]->tops[0];
    // out
    net->layers[40]->p_bottoms[0] = &net->layers[38]->tops[0];
    net->layers[40]->p_bottoms[1] = &net->layers[39]->tops[0];
    net->layers[40]->p_bottoms[2] = &net->layers[30]->tops[0];
    net->layers[40]->p_bottoms[3] = net->img_info;
  }
}

void shape_frcnn_7_1_1(Net* const net)
{
  // PVANET
  {
    // conv1_1, ..., conv3_3
    for (int i = 1; i <= 7; ++i) {
      shape_conv_layer(net, net->layers[i]);
    }
    // downsample
    shape_pool_layer(net, net->layers[8]);
    // conv4_1, ..., conv5_3
    for (int i = 9; i <= 14; ++i) {
      shape_conv_layer(net, net->layers[i]);
    }
    // upsample
    shape_deconv_layer(net, net->layers[15]);
    // concat
    shape_concat_layer(net, net->layers[16]);
    // convf
    shape_conv_layer(net, net->layers[17]);
  }

  // Multi-scale RPN
  {
    // rpn_conv1, 3, 5, rpn_cls_score1, 3, 5, rpn_bbox_pred1, 3, 5
    for (int i = 18; i <= 26; ++i) {
      shape_conv_layer(net, net->layers[i]);
    }
    // rpn_score
    shape_concat_layer(net, net->layers[27]);
    // rpn_pred
    {
      Tensor* const pred = &net->layers[28]->tops[0];
      const Tensor* const score = net->layers[28]->p_bottoms[0];
      pred->ndim = 4;
      pred->num_items = score->num_items;
      for (int n = 0; n < score->num_items; ++n) {
        pred->shape[n][0] = 2;
        pred->shape[n][1] = score->shape[n][0] / 2;
        pred->shape[n][2] = score->shape[n][1];
        pred->shape[n][3] = score->shape[n][2];
      }
    }
    // rpn_bbox
    shape_concat_layer(net, net->layers[29]);
    {
      Tensor* const bbox = &net->layers[29]->tops[0];
      bbox->ndim = 4;
      for (int n = 0; n < bbox->num_items; ++n) {
        const int C = bbox->shape[n][0];
        const int H = bbox->shape[n][1];
        const int W = bbox->shape[n][2];
        bbox->shape[n][0] = C / 4;
        bbox->shape[n][1] = 4;
        bbox->shape[n][2] = H;
        bbox->shape[n][3] = W;
      }
    }
    // img_info
    {
      Tensor* const bbox = &net->layers[29]->tops[0];
      net->img_info->ndim = 1;
      net->img_info->num_items = bbox->num_items;
      for (int n = 0; n < net->img_info->num_items; ++n) {
        net->img_info->shape[n][0] = 4;
        net->img_info->start[n] = n * 4;
      }
    }
    // proposal
    shape_proposal_layer(net, net->layers[30]);
  }

  // R-CNN
  {
    // roipool
    shape_roipool_layer(net, net->layers[31]);
    // roipool reshape
    {
      Tensor* const roipool_flat = &net->layers[32]->tops[0];
      const Tensor* const roipool = &net->layers[31]->tops[0];
      // calculate total number of RoI-pooled data
      int total_num_rois = 0;
      for (int n = 0; n < roipool->num_items; ++n) {
        total_num_rois += roipool->shape[n][0];
      }

      // reshape to 2d tensor: total_num_rois x (C * H * W)
      roipool_flat->ndim = 2;
      roipool_flat->num_items = 1;
      roipool_flat->shape[0][0] = total_num_rois;
      roipool_flat->shape[0][1] = roipool->shape[0][1]
                                  * roipool->shape[0][2]
                                  * roipool->shape[0][3];
      roipool_flat->start[0] = 0;
    }
    // fc6_L, 6_U, 7_L, 7_U
    for (int i = 33; i <= 36; ++i) {
      shape_fc_layer(net, net->layers[i]);
    }
    // score
    shape_fc_layer(net, net->layers[37]);
    // pred
    {
      Tensor* const pred = &net->layers[38]->tops[0];
      const Tensor* const roipool = &net->layers[31]->tops[0];
      pred->ndim = 2;
      pred->num_items = roipool->num_items;
      for (int n = 0; n < roipool->num_items; ++n) {
        pred->shape[n][0] = roipool->shape[n][0];
        pred->shape[n][1] = net->layers[37]->option.out_channels;
      }
    }
    // bbox
    shape_fc_layer(net, net->layers[39]);
    {
      Tensor* const bbox = &net->layers[39]->tops[0];
      const Tensor* const roipool = &net->layers[31]->tops[0];
      bbox->ndim = 3;
      bbox->num_items = roipool->num_items;
      for (int n = 0; n < roipool->num_items; ++n) {
        bbox->shape[n][0] = roipool->shape[n][0];
        bbox->shape[n][1] = net->layers[39]->option.out_channels / 4;
        bbox->shape[n][2] = 4;
      }
    }
    // out
    shape_odout_layer(net, net->layers[40]);
  }
}

void construct_frcnn_7_1_1(Net* net)
{
  long int space_cpu = 0;

  setup_frcnn_7_1_1(net);

  for (int i = 0; i < net->num_layers; ++i) {
    space_cpu += malloc_layer(net->layers[i]);
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

  connect_frcnn_7_1_1(net);

  Tensor* input = &net->layers[0]->tops[0];
  input->num_items = 4;
  input->ndim = 3;
  for (int n = 0; n < input->num_items; ++n) {
    input->shape[n][0] = 3;
    input->shape[n][1] = 640;
    input->shape[n][2] = 1024;
    input->start[n] = n * 3 * 640 * 1024;
  }
  shape_frcnn_7_1_1(net);

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
    net->layers[28]->tops[0].data = net->layers[27]->tops[0].data;
    net->layers[29]->tops[0].data = net->layer_data[2];
    net->layers[31]->tops[0].data = net->layer_data[2];
    net->layers[32]->tops[0].data = net->layers[31]->tops[0].data;
    net->layers[34]->tops[0].data = net->layer_data[1];
    net->layers[36]->tops[0].data = net->layer_data[1];
    net->layers[37]->tops[0].data = net->layer_data[0];
    net->layers[38]->tops[0].data = net->layers[37]->tops[0].data;
    net->layers[39]->tops[0].data = net->layer_data[2];
  }

  const int num_anchors = net->layers[30]->option.num_scales
                          * net->layers[30]->option.num_ratios
                          * net->layers[30]->option.num_concats;
  #ifdef GPU
  {
    cudaMalloc(&net->anchors, num_anchors * 4 * sizeof(real));
    generate_anchors(net->param_cpu_data, &net->layers[30]->option);
    cudaMemcpyAsync(net->anchors, net->param_cpu_data,
                    num_anchors * 4 * sizeof(real),
                    cudaMemcpyHostToDevice);
  }
  #else
  {
    net->anchors = (real*)malloc(num_anchors * 4 * sizeof(real));
    generate_anchors(net->anchors, &net->layers[30]->option);
  }
  #endif
  net->space += num_anchors * 4 * sizeof(real);

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
  shape_frcnn_7_1_1(net);

  print_tensor_info("data", input);
  print_tensor_info("img_info", net->img_info);
}

void get_output(Net* net, const int image_start_index, FILE* fp)
{
  // retrieve output
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
  }

  // print output
  {
    const Tensor* const out = &net->layers[net->num_layers - 1]->tops[0];

    for (int n = 0; n < out->num_items; ++n) {
      const int image_index = image_start_index + n;
      const real* const p_out_item = net->output_cpu_data + out->start[n];
      for (int i = 0; i < out->shape[n][0]; ++i) {
        const int class_index = (int)p_out_item[i * 6 + 0];
        printf("Image %d / Box %d: class %d, score %f, p1 = (%.2f, %.2f), p2 = (%.2f, %.2f)\n",
               image_index, i, class_index, p_out_item[i * 6 + 5],
               p_out_item[i * 6 + 1], p_out_item[i * 6 + 2],
               p_out_item[i * 6 + 3], p_out_item[i * 6 + 4]);

        fwrite(&image_index, sizeof(int), 1, fp);
        fwrite(&class_index, sizeof(int), 1, fp);
        fwrite(&p_out_item[i * 6 + 1], sizeof(real), 5, fp);
      }
    }
  }
}

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
        forward_frcnn_7_1_1(&frcnn);

        // retrieve output & save to file
        get_output(&frcnn, total_count, fp_out);

        total_count += count;
        count = 0;
        buf_count = 0;
      }
    }

    if (count > 0) {
      prepare_input(&frcnn, (const char * const *)&line, count);
      forward_frcnn_7_1_1(&frcnn);
      get_output(&frcnn, total_count, fp_out);
    }

    fclose(fp_list);
    fclose(fp_out);
  }


  // end
  free_net(&frcnn);

  return 0;
}
