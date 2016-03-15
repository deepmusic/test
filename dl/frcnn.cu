#include "layer.h"
#include <stdio.h>
#include <string.h>

typedef struct PVANET_
{
  Tensor input;
  Tensor conv1_1, conv1_2;
  Tensor weight1_1, bias1_1, weight1_2, bias1_2;
  Tensor conv2_1, conv2_2;
  Tensor weight2_1, bias2_1, weight2_2, bias2_2;
  Tensor conv3_1, conv3_2, conv3_3;
  Tensor weight3_1, bias3_1, weight3_2, bias3_2, weight3_3, bias3_3;
  Tensor conv4_1, conv4_2, conv4_3;
  Tensor weight4_1, bias4_1, weight4_2, bias4_2, weight4_3, bias4_3;
  Tensor conv5_1, conv5_2, conv5_3;
  Tensor weight5_1, bias5_1, weight5_2, bias5_2, weight5_3, bias5_3;
  Tensor downsample, upsample, concat;
  Tensor weight_up, bias_up;
  Tensor convf;
  Tensor weightf, biasf;
} PVANET;

typedef struct SRPN_
{
  Tensor conv1, conv3, conv5;
  Tensor weight_c1, bias_c1, weight_c3, bias_c3, weight_c5, bias_c5;
  Tensor score1, score3, score5;
  Tensor weight_s1, bias_s1, weight_s3, bias_s3, weight_s5, bias_s5;
  Tensor bbox1, bbox3, bbox5;
  Tensor weight_b1, bias_b1, weight_b3, bias_b3, weight_b5, bias_b5;
  Tensor score, pred, bbox;
  Tensor img_info;
  Tensor roi;
} SRPN;

typedef struct RCNN_
{
  Tensor roipool, roipool_flat;
  Tensor fc6_1, fc6_2;
  Tensor weight6_1, bias6_1, weight6_2, bias6_2;
  Tensor fc7_1, fc7_2;
  Tensor weight7_1, bias7_1, weight7_2, bias7_2;
  Tensor score, bbox, pred;
  Tensor weight_s, bias_s, weight_b, bias_b;
  Tensor out;
} RCNN;

PVANET pvanet;
SRPN srpn;
RCNN rcnn;

ConvOption conv_option1;
ConvOption conv_option2;
ConvOption conv1x1_option;
ConvOption conv5x5_option;
ConvOption deconv_option;
PoolOption pool_option;
ReluOption relu_option;
DropoutOption dropout_option;
ProposalOption proposal_option;
ROIPoolOption roipool_option;
FCOption fc_option;
ODOutOption odout_option;

const Tensor* const concat_bottoms[3]
    = { &pvanet.downsample, &pvanet.conv4_3, &pvanet.upsample };
const Tensor* const score_bottoms[3]
    = { &srpn.score1, &srpn.score3, &srpn.score5 };
const Tensor* const bbox_bottoms[3]
    = { &srpn.bbox1, &srpn.bbox3, &srpn.bbox5 };

int max_layer_size = 0, max_param_size = 0;
int max_temp_size = 0, max_const_size = 0, max_tempint_size = 0;

real* layer1_data = NULL;
real* layer2_data = NULL;
real* layer3_data = NULL;
real* backup1_data = NULL;
real* backup2_data = NULL;
real* temp_data = NULL;
real* const_data = NULL;
int* tempint_data = NULL;

real* true_data = NULL;
real* input_data = NULL;
real* output_data = NULL;
real* param_data = NULL;

real* anchors = NULL;
real anchor_scales[5] = { 3.0f, 6.0f, 9.0f, 16.0f, 32.0f };
real anchor_ratios[5] = { 0.5f, 0.666f, 1.0f, 1.5f, 2.0f };
real* proposal_temp = NULL;
int* proposal_tempint = NULL;

void forward_frcnn_7_1_1(void)
{
  // PVANET
  {
    // 1_1
    pvanet.input.data = layer1_data;
    pvanet.conv1_1.data = layer2_data;
    conv_option1.out_channels = 32;
    conv_forward(&pvanet.input, &pvanet.conv1_1,
                &pvanet.weight1_1, &pvanet.bias1_1,
                temp_data, const_data, &conv_option1);
    relu_forward_inplace(&pvanet.conv1_1, &relu_option);
    print_tensor_info("input", &pvanet.input);
    print_tensor_info("conv1_1", &pvanet.conv1_1);

    // 1_2
    pvanet.conv1_2.data = layer1_data;
    conv_option2.out_channels = 32;
    conv_forward(&pvanet.conv1_1, &pvanet.conv1_2,
                 &pvanet.weight1_2, &pvanet.bias1_2,
                 temp_data, const_data, &conv_option2);
    relu_forward_inplace(&pvanet.conv1_2, &relu_option);
    print_tensor_info("conv1_2", &pvanet.conv1_2);

    // 2_1
    pvanet.conv2_1.data = layer2_data;
    conv_option1.out_channels = 64;
    conv_forward(&pvanet.conv1_2, &pvanet.conv2_1,
                 &pvanet.weight2_1, &pvanet.bias2_1,
                 temp_data, const_data, &conv_option1);
    relu_forward_inplace(&pvanet.conv2_1, &relu_option);
    print_tensor_info("conv2_1", &pvanet.conv2_1);

    // 2_2
    pvanet.conv2_2.data = layer1_data;
    conv_option2.out_channels = 64;
    conv_forward(&pvanet.conv2_1, &pvanet.conv2_2,
                 &pvanet.weight2_2, &pvanet.bias2_2,
                 temp_data, const_data, &conv_option2);
    relu_forward_inplace(&pvanet.conv2_2, &relu_option);
    print_tensor_info("conv2_2", &pvanet.conv2_2);

    // 3_1
    pvanet.conv3_1.data = layer2_data;
    conv_option1.out_channels = 96;
    conv_forward(&pvanet.conv2_2, &pvanet.conv3_1,
                 &pvanet.weight3_1, &pvanet.bias3_1,
                 temp_data, const_data, &conv_option1);
    relu_forward_inplace(&pvanet.conv3_1, &relu_option);
    print_tensor_info("conv3_1", &pvanet.conv3_1);

    // 3_2
    pvanet.conv3_2.data = layer1_data;
    conv_option2.out_channels = 64;
    conv_forward(&pvanet.conv3_1, &pvanet.conv3_2,
                 &pvanet.weight3_2, &pvanet.bias3_2,
                 temp_data, const_data, &conv_option2);
    relu_forward_inplace(&pvanet.conv3_2, &relu_option);
    print_tensor_info("conv3_2", &pvanet.conv3_2);

    // 3_3
    pvanet.conv3_3.data = layer2_data;
    conv_option2.out_channels = 128;
    conv_forward(&pvanet.conv3_2, &pvanet.conv3_3,
                 &pvanet.weight3_3, &pvanet.bias3_3,
                 temp_data, const_data, &conv_option2);
    relu_forward_inplace(&pvanet.conv3_3, &relu_option);
    print_tensor_info("conv3_3", &pvanet.conv3_3);

    // downsample
    pvanet.downsample.data = backup1_data;
    pool_forward(&pvanet.conv3_3, &pvanet.downsample,
                 tempint_data, &pool_option);
    print_tensor_info("downsample", &pvanet.downsample);

    // 4_1
    pvanet.conv4_1.data = layer1_data;
    conv_option1.out_channels = 192;
    conv_forward(&pvanet.conv3_3, &pvanet.conv4_1,
                 &pvanet.weight4_1, &pvanet.bias4_1,
                 temp_data, const_data, &conv_option1);
    relu_forward_inplace(&pvanet.conv4_1, &relu_option);
    print_tensor_info("conv4_1", &pvanet.conv4_1);

    // 4_2
    pvanet.conv4_2.data = layer2_data;
    conv_option2.out_channels = 128;
    conv_forward(&pvanet.conv4_1, &pvanet.conv4_2,
                 &pvanet.weight4_2, &pvanet.bias4_2,
                 temp_data, const_data, &conv_option2);
    relu_forward_inplace(&pvanet.conv4_2, &relu_option);
    print_tensor_info("conv4_2", &pvanet.conv4_2);

    // 4_3
    pvanet.conv4_3.data = backup2_data;
    conv_option2.out_channels = 256;
    conv_forward(&pvanet.conv4_2, &pvanet.conv4_3,
                 &pvanet.weight4_3, &pvanet.bias4_3,
                 temp_data, const_data, &conv_option2);
    relu_forward_inplace(&pvanet.conv4_3, &relu_option);
    print_tensor_info("conv4_3", &pvanet.conv4_3);

    // 5_1
    pvanet.conv5_1.data = layer1_data;
    conv_option1.out_channels = 384;
    conv_forward(&pvanet.conv4_3, &pvanet.conv5_1,
                 &pvanet.weight5_1, &pvanet.bias5_1,
                 temp_data, const_data, &conv_option1);
    relu_forward_inplace(&pvanet.conv5_1, &relu_option);
    print_tensor_info("conv5_1", &pvanet.conv5_1);

    // 5_2
    pvanet.conv5_2.data = layer2_data;
    conv_option2.out_channels = 256;
    conv_forward(&pvanet.conv5_1, &pvanet.conv5_2,
                 &pvanet.weight5_2, &pvanet.bias5_2,
                 temp_data, const_data, &conv_option2);
    relu_forward_inplace(&pvanet.conv5_2, &relu_option);
    print_tensor_info("conv5_2", &pvanet.conv5_2);

    // 5_3
    pvanet.conv5_3.data = layer1_data;
    conv_option2.out_channels = 512;
    conv_forward(&pvanet.conv5_2, &pvanet.conv5_3,
                 &pvanet.weight5_3, &pvanet.bias5_3,
                 temp_data, const_data, &conv_option2);
    relu_forward_inplace(&pvanet.conv5_3, &relu_option);
    print_tensor_info("conv5_3", &pvanet.conv5_3);

    // upsample
    pvanet.upsample.data = layer2_data;
    deconv_forward(&pvanet.conv5_3, &pvanet.upsample,
                   &pvanet.weight_up, &pvanet.bias_up,
                   temp_data, const_data, &deconv_option);
    print_tensor_info("upsample", &pvanet.upsample);

    // concat
    pvanet.concat.data = layer1_data;
    concat_forward(concat_bottoms, &pvanet.concat, 3);
    print_tensor_info("concat", &pvanet.concat);

    // convf
    pvanet.convf.data = backup1_data;
    conv1x1_option.out_channels = 512;
    conv_forward(&pvanet.concat, &pvanet.convf,
                 &pvanet.weightf, &pvanet.biasf,
                 temp_data, const_data, &conv1x1_option);
    relu_forward_inplace(&pvanet.convf, &relu_option);
    print_tensor_info("convf", &pvanet.convf);
  }

  // SRPN
  {
    // conv1
    srpn.conv1.data = layer1_data;
    conv1x1_option.out_channels = 128;
    conv_forward(&pvanet.convf, &srpn.conv1,
                 &srpn.weight_c1, &srpn.bias_c1,
                 temp_data, const_data, &conv1x1_option);
    relu_forward_inplace(&srpn.conv1, &relu_option);
    print_tensor_info("rpn_conv1", &srpn.conv1);

    // conv3
    srpn.conv3.data = layer2_data;
    conv_option2.out_channels = 256;
    conv_forward(&pvanet.convf, &srpn.conv3,
                 &srpn.weight_c3, &srpn.bias_c3,
                 temp_data, const_data, &conv_option2);
    relu_forward_inplace(&srpn.conv3, &relu_option);
    print_tensor_info("rpn_conv3", &srpn.conv3);

    // conv5
    srpn.conv5.data = layer3_data;
    conv5x5_option.out_channels = 128;
    conv_forward(&pvanet.convf, &srpn.conv5,
                 &srpn.weight_c5, &srpn.bias_c5,
                 temp_data, const_data, &conv5x5_option);
    relu_forward_inplace(&srpn.conv5, &relu_option);
    print_tensor_info("rpn_conv5", &srpn.conv5);

    // score1
    conv1x1_option.out_channels = 50;
    conv_forward(&srpn.conv1, &srpn.score1,
                 &srpn.weight_s1, &srpn.bias_s1,
                 temp_data, const_data, &conv1x1_option);
    print_tensor_info("rpn_score1", &srpn.score1);

    // score3
    conv1x1_option.out_channels = 50;
    conv_forward(&srpn.conv3, &srpn.score3,
                 &srpn.weight_s3, &srpn.bias_s3,
                 temp_data, const_data, &conv1x1_option);
    print_tensor_info("rpn_score3", &srpn.score3);

    // score5
    conv1x1_option.out_channels = 50;
    conv_forward(&srpn.conv5, &srpn.score5,
                 &srpn.weight_s5, &srpn.bias_s5,
                 temp_data, const_data, &conv1x1_option);
    print_tensor_info("rpn_score5", &srpn.score5);

    // bbox1
    conv1x1_option.out_channels = 100;
    conv_forward(&srpn.conv1, &srpn.bbox1,
                 &srpn.weight_b1, &srpn.bias_b1,
                 temp_data, const_data, &conv1x1_option);
    print_tensor_info("rpn_bbox1", &srpn.bbox1);

    // bbox3
    conv1x1_option.out_channels = 100;
    conv_forward(&srpn.conv3, &srpn.bbox3,
                 &srpn.weight_b3, &srpn.bias_b3,
                 temp_data, const_data, &conv1x1_option);
    print_tensor_info("rpn_bbox3", &srpn.bbox3);

    // bbox5
    conv1x1_option.out_channels = 100;
    conv_forward(&srpn.conv5, &srpn.bbox5,
                 &srpn.weight_b5, &srpn.bias_b5,
                 temp_data, const_data, &conv1x1_option);
    print_tensor_info("rpn_bbox5", &srpn.bbox5);

    // score
    srpn.score.data = layer1_data;
    concat_forward(score_bottoms, &srpn.score, 3);
    print_tensor_info("rpn_score", &srpn.score);

    // pred
    srpn.pred.ndim = 3;
    srpn.pred.num_items = srpn.score.num_items;
    for (int n = 0; n < srpn.score.num_items; ++n) {
      srpn.pred.shape[n][0] = 2;
      srpn.pred.shape[n][1]
          = srpn.score.shape[n][0] / 2 * srpn.score.shape[n][1];
      srpn.pred.shape[n][2] = srpn.score.shape[n][2];
      srpn.pred.start[n] = srpn.score.start[n];
    }
    srpn.pred.data = srpn.score.data;
    softmax_inplace_forward(&srpn.pred, temp_data);
    print_tensor_info("rpn_pred", &srpn.pred);

    // pred reshape
    srpn.pred.ndim = 4;
    srpn.pred.num_items = srpn.score.num_items;
    for (int n = 0; n < srpn.score.num_items; ++n) {
      srpn.pred.shape[n][0] = 2;
      srpn.pred.shape[n][1] = srpn.score.shape[n][0] / 2;
      srpn.pred.shape[n][2] = srpn.score.shape[n][1];
      srpn.pred.shape[n][3] = srpn.score.shape[n][2];
    }
    print_tensor_info("rpn_pred_reshape", &srpn.pred);

    // bbox
    srpn.bbox.data = layer2_data;
    concat_forward(bbox_bottoms, &srpn.bbox, 3);
    print_tensor_info("rpn_bbox", &srpn.bbox);
    // bbox reshape
    srpn.bbox.ndim = 4;
    for (int n = 0; n < srpn.bbox.num_items; ++n) {
      const int C = srpn.bbox.shape[n][0];
      const int H = srpn.bbox.shape[n][1];
      const int W = srpn.bbox.shape[n][2];
      srpn.bbox.shape[n][0] = C / 4;
      srpn.bbox.shape[n][1] = 4;
      srpn.bbox.shape[n][2] = H;
      srpn.bbox.shape[n][3] = W;
    }
    print_tensor_info("rpn_bbox_reshape", &srpn.bbox);

    // proposal
    proposal_forward(&srpn.pred, &srpn.bbox, &srpn.img_info,
                     &srpn.roi, anchors,
                     proposal_temp, proposal_tempint,
                     temp_data, tempint_data,
                     &proposal_option);
    print_tensor_info("img_info", &srpn.img_info);
    print_tensor_info("roi", &srpn.roi);
  }

  // R-CNN
  {
    // roipool
    rcnn.roipool.data = layer1_data;
    roipool_forward(&pvanet.convf, &srpn.roi, &rcnn.roipool,
                    tempint_data, &roipool_option);
    print_tensor_info("roipool", &rcnn.roipool);

    // roipool reshape
    {
      // calculate total number of RoI-pooled data
      int total_num_rois = 0;
      for (int n = 0; n < rcnn.roipool.num_items; ++n) {
        total_num_rois += rcnn.roipool.shape[n][0];
      }

      // reshape to 2d tensor: total_num_rois x (C * H * W)
      rcnn.roipool_flat.ndim = 2;
      rcnn.roipool_flat.num_items = 1;
      rcnn.roipool_flat.shape[0][0] = total_num_rois;
      rcnn.roipool_flat.shape[0][1] = rcnn.roipool.shape[0][1]
                                      * rcnn.roipool.shape[0][2]
                                      * rcnn.roipool.shape[0][3];
      rcnn.roipool_flat.start[0] = 0;
    }
    print_tensor_info("roipool_flat", &rcnn.roipool_flat);

    // fc6
    rcnn.roipool_flat.data = rcnn.roipool.data;
    rcnn.fc6_1.data = layer2_data;
    fc_option.out_channels = 512;
    fc_option.bias = 0;
    fc_forward(&rcnn.roipool_flat, &rcnn.fc6_1,
               &rcnn.weight6_1, &rcnn.bias6_1,
               const_data, &fc_option);
    print_tensor_info("fc6_1", &rcnn.fc6_1);

    rcnn.fc6_2.data = layer1_data;
    fc_option.out_channels = 4096;
    fc_option.bias = 1;
    fc_forward(&rcnn.fc6_1, &rcnn.fc6_2, &rcnn.weight6_2, &rcnn.bias6_2,
               const_data, &fc_option);

    relu_forward_inplace(&rcnn.fc6_2, &relu_option);
    dropout_forward_inplace(&rcnn.fc6_2, NULL, &dropout_option);
    print_tensor_info("fc6_2", &rcnn.fc6_2);

    // fc7
    rcnn.fc7_1.data = backup2_data;
    fc_option.out_channels = 4096;
    fc_option.bias = 1;
    fc_forward(&rcnn.fc6_2, &rcnn.fc7_1, &rcnn.weight7_1, &rcnn.bias7_2,
               const_data, &fc_option);
    print_tensor_info("fc7_1", &rcnn.fc7_1);
/*
    rcnn.fc7_2.data = layer1_data;
    fc_option.out_channels = 4096;
    fc_option.bias = 1;
    fc_forward(&rcnn.fc7_1, &rcnn.fc7_2, &rcnn.weight7_2, &rcnn.bias7_2,
               const_data, &fc_option);
*/
    relu_forward_inplace(&rcnn.fc7_1, &relu_option);
    dropout_forward_inplace(&rcnn.fc7_1, NULL, &dropout_option);
    print_tensor_info("fc7_2", &rcnn.fc7_2);

    // bbox
    rcnn.bbox.data = layer2_data;
    fc_option.out_channels = 84;
    fc_forward(&rcnn.fc7_1, &rcnn.bbox, &rcnn.weight_b, &rcnn.bias_b,
               const_data, &fc_option);
    print_tensor_info("bbox", &rcnn.bbox);
    // bbox reshape
    rcnn.bbox.ndim = 3;
    rcnn.bbox.num_items = rcnn.roipool.num_items;
    {
      int total_size = 0;
      for (int n = 0; n < rcnn.roipool.num_items; ++n) {
        const int item_size = rcnn.roipool.shape[n][0] * 21 * 4;
        rcnn.bbox.shape[n][0] = rcnn.roipool.shape[n][0];
        rcnn.bbox.shape[n][1] = 21;
        rcnn.bbox.shape[n][2] = 4;
        rcnn.bbox.start[n] = total_size;
        total_size += item_size;
      }
    }

    // score
    rcnn.score.data = backup1_data;
    fc_option.out_channels = 21;
    fc_forward(&rcnn.fc7_1, &rcnn.score, &rcnn.weight_s, &rcnn.bias_s,
               const_data, &fc_option);
    print_tensor_info("score", &rcnn.score);

    // pred
    rcnn.pred.ndim = 4;
    rcnn.pred.num_items = 1;
    rcnn.pred.shape[0][0] = rcnn.roipool_flat.shape[0][0];
    rcnn.pred.shape[0][1] = 21;
    rcnn.pred.shape[0][2] = 1;
    rcnn.pred.shape[0][3] = 1;
    rcnn.pred.start[0] = 0;
    rcnn.pred.data = rcnn.score.data;
    softmax_inplace_forward(&rcnn.pred, temp_data);
    print_tensor_info("pred", &rcnn.pred);

    // pred reshape
    rcnn.pred.ndim = 2;
    rcnn.pred.num_items = rcnn.roipool.num_items;
    {
      int total_size = 0;
      for (int n = 0; n < rcnn.roipool.num_items; ++n) {
        const int item_size = rcnn.roipool.shape[n][0] * 21;
        rcnn.pred.shape[n][0] = rcnn.roipool.shape[n][0];
        rcnn.pred.shape[n][1] = 21;
        rcnn.pred.start[n] = total_size;
        total_size += item_size;
      }
    }
    print_tensor_info("pred", &rcnn.pred);

    rcnn.out.data = backup2_data;
    odout_forward(&rcnn.pred, &rcnn.bbox, &srpn.roi, &srpn.img_info,
                  &rcnn.out,
                  proposal_temp, proposal_tempint,
                  temp_data, tempint_data,
                  &odout_option);
    print_tensor_info("out", &rcnn.out);
  }
}

void shape_frcnn_7_1_1(const int print_network_info)
{
  int temp_size, const_size, tempint_size;

  // PVANET
  {
    // 1_1
    conv_option1.out_channels = 32;
    conv_shape(&pvanet.input, &pvanet.conv1_1,
               &pvanet.weight1_1, &pvanet.bias1_1,
               &temp_size, &const_size, &conv_option1);
    max_layer_size = MAX(max_layer_size,  flatten_size(&pvanet.input));
    max_layer_size = MAX(max_layer_size,  flatten_size(&pvanet.conv1_1));
    max_param_size = MAX(max_param_size,  flatten_size(&pvanet.weight1_1));
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_const_size = MAX(max_const_size,  const_size);

    // 1_2
    conv_option2.out_channels = 32;
    conv_shape(&pvanet.conv1_1, &pvanet.conv1_2,
               &pvanet.weight1_2, &pvanet.bias1_2,
               &temp_size, &const_size, &conv_option2);
    max_layer_size = MAX(max_layer_size,  flatten_size(&pvanet.conv1_2));
    max_param_size = MAX(max_param_size,  flatten_size(&pvanet.weight1_2));
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_const_size = MAX(max_const_size,  const_size);

    // 2_1
    conv_option1.out_channels = 64;
    conv_shape(&pvanet.conv1_2, &pvanet.conv2_1,
               &pvanet.weight2_1, &pvanet.bias2_1,
               &temp_size, &const_size, &conv_option1);
    max_layer_size = MAX(max_layer_size,  flatten_size(&pvanet.conv2_1));
    max_param_size = MAX(max_param_size,  flatten_size(&pvanet.weight2_1));
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_const_size = MAX(max_const_size,  const_size);

    // 2_2
    conv_option2.out_channels = 64;
    conv_shape(&pvanet.conv2_1, &pvanet.conv2_2,
               &pvanet.weight2_2, &pvanet.bias2_2,
               &temp_size, &const_size, &conv_option2);
    max_layer_size = MAX(max_layer_size,  flatten_size(&pvanet.conv2_2));
    max_param_size = MAX(max_param_size,  flatten_size(&pvanet.weight2_2));
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_const_size = MAX(max_const_size,  const_size);

    // 3_1
    conv_option1.out_channels = 96;
    conv_shape(&pvanet.conv2_2, &pvanet.conv3_1,
               &pvanet.weight3_1, &pvanet.bias3_1,
               &temp_size, &const_size, &conv_option1);
    max_layer_size = MAX(max_layer_size,  flatten_size(&pvanet.conv3_1));
    max_param_size = MAX(max_param_size,  flatten_size(&pvanet.weight3_1));
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_const_size = MAX(max_const_size,  const_size);

    // 3_2
    conv_option2.out_channels = 64;
    conv_shape(&pvanet.conv3_1, &pvanet.conv3_2,
               &pvanet.weight3_2, &pvanet.bias3_2,
               &temp_size, &const_size, &conv_option2);
    max_layer_size = MAX(max_layer_size,  flatten_size(&pvanet.conv3_2));
    max_param_size = MAX(max_param_size,  flatten_size(&pvanet.weight3_2));
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_const_size = MAX(max_const_size,  const_size);

    // 3_3
    conv_option2.out_channels = 128;
    conv_shape(&pvanet.conv3_2, &pvanet.conv3_3,
               &pvanet.weight3_3, &pvanet.bias3_3,
               &temp_size, &const_size, &conv_option2);
    max_layer_size = MAX(max_layer_size,  flatten_size(&pvanet.conv3_3));
    max_param_size = MAX(max_param_size,  flatten_size(&pvanet.weight3_3));
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_const_size = MAX(max_const_size,  const_size);

    // 4_1
    conv_option1.out_channels = 192;
    conv_shape(&pvanet.conv3_3, &pvanet.conv4_1,
               &pvanet.weight4_1, &pvanet.bias4_1,
               &temp_size, &const_size, &conv_option1);
    max_layer_size = MAX(max_layer_size,  flatten_size(&pvanet.conv4_1));
    max_param_size = MAX(max_param_size,  flatten_size(&pvanet.weight4_1));
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_const_size = MAX(max_const_size,  const_size);

    // 4_2
    conv_option2.out_channels = 128;
    conv_shape(&pvanet.conv4_1, &pvanet.conv4_2,
               &pvanet.weight4_2, &pvanet.bias4_2,
               &temp_size, &const_size, &conv_option2);
    max_layer_size = MAX(max_layer_size,  flatten_size(&pvanet.conv4_2));
    max_param_size = MAX(max_param_size,  flatten_size(&pvanet.weight4_2));
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_const_size = MAX(max_const_size,  const_size);

    // 4_3
    conv_option2.out_channels = 256;
    conv_shape(&pvanet.conv4_2, &pvanet.conv4_3,
               &pvanet.weight4_3, &pvanet.bias4_3,
               &temp_size, &const_size, &conv_option2);
    max_layer_size = MAX(max_layer_size,  flatten_size(&pvanet.conv4_3));
    max_param_size = MAX(max_param_size,  flatten_size(&pvanet.weight4_3));
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_const_size = MAX(max_const_size,  const_size);

    // 5_1
    conv_option1.out_channels = 384;
    conv_shape(&pvanet.conv4_3, &pvanet.conv5_1,
               &pvanet.weight5_1, &pvanet.bias5_1,
               &temp_size, &const_size, &conv_option1);
    max_layer_size = MAX(max_layer_size,  flatten_size(&pvanet.conv5_1));
    max_param_size = MAX(max_param_size,  flatten_size(&pvanet.weight5_1));
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_const_size = MAX(max_const_size,  const_size);

    // 5_2
    conv_option2.out_channels = 256;
    conv_shape(&pvanet.conv5_1, &pvanet.conv5_2,
               &pvanet.weight5_2, &pvanet.bias5_2,
               &temp_size, &const_size, &conv_option2);
    max_layer_size = MAX(max_layer_size,  flatten_size(&pvanet.conv5_2));
    max_param_size = MAX(max_param_size,  flatten_size(&pvanet.weight5_2));
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_const_size = MAX(max_const_size,  const_size);

    // 5_3
    conv_option2.out_channels = 512;
    conv_shape(&pvanet.conv5_2, &pvanet.conv5_3,
               &pvanet.weight5_3, &pvanet.bias5_3,
               &temp_size, &const_size, &conv_option2);
    max_layer_size = MAX(max_layer_size,  flatten_size(&pvanet.conv5_3));
    max_param_size = MAX(max_param_size,  flatten_size(&pvanet.weight5_3));
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_const_size = MAX(max_const_size,  const_size);

    // downsample
    pool_shape(&pvanet.conv3_3, &pvanet.downsample,
               &tempint_size, &pool_option);
    max_layer_size = MAX(max_layer_size,  flatten_size(&pvanet.downsample));
    max_tempint_size = MAX(max_tempint_size,  tempint_size);

    // upsample
    deconv_shape(&pvanet.conv5_3, &pvanet.upsample,
                 &pvanet.weight_up, &pvanet.bias_up,
                 &temp_size, &const_size, &deconv_option);
    max_layer_size = MAX(max_layer_size,  flatten_size(&pvanet.upsample));
    max_param_size = MAX(max_param_size,  flatten_size(&pvanet.weight_up));
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_const_size = MAX(max_const_size,  const_size);

    // concat
    concat_shape(concat_bottoms, &pvanet.concat, 3);
    max_layer_size = MAX(max_layer_size,  flatten_size(&pvanet.concat));

    // convf
    conv1x1_option.out_channels = 512;
    conv_shape(&pvanet.concat, &pvanet.convf,
               &pvanet.weightf, &pvanet.biasf,
               &temp_size, &const_size, &conv1x1_option);
    max_layer_size = MAX(max_layer_size,  flatten_size(&pvanet.convf));
    max_param_size = MAX(max_param_size,  flatten_size(&pvanet.weightf));
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_const_size = MAX(max_const_size,  const_size);
  }

  if (print_network_info) {
    print_tensor_info("input", &pvanet.input);
    print_tensor_info("conv1_1", &pvanet.conv1_1);
    print_tensor_info("conv1_2", &pvanet.conv1_2);
    print_tensor_info("conv2_1", &pvanet.conv2_1);
    print_tensor_info("conv2_2", &pvanet.conv2_2);
    print_tensor_info("conv3_1", &pvanet.conv3_1);
    print_tensor_info("conv3_2", &pvanet.conv3_2);
    print_tensor_info("conv3_3", &pvanet.conv3_3);
    print_tensor_info("conv4_1", &pvanet.conv4_1);
    print_tensor_info("conv4_2", &pvanet.conv4_2);
    print_tensor_info("conv4_3", &pvanet.conv4_3);
    print_tensor_info("conv5_1", &pvanet.conv5_1);
    print_tensor_info("conv5_2", &pvanet.conv5_2);
    print_tensor_info("conv5_3", &pvanet.conv5_3);
    print_tensor_info("downsample", &pvanet.downsample);
    print_tensor_info("upsample", &pvanet.upsample);
    print_tensor_info("concat", &pvanet.concat);
    print_tensor_info("convf", &pvanet.convf);
  }

  // SRPN
  {
    // conv1
    conv1x1_option.out_channels = 128;
    conv_shape(&pvanet.convf, &srpn.conv1,
               &srpn.weight_c1, &srpn.bias_c1,
               &temp_size, &const_size, &conv1x1_option);
    max_layer_size = MAX(max_layer_size,  flatten_size(&srpn.conv1));
    max_param_size = MAX(max_param_size,  flatten_size(&srpn.weight_c1));
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_const_size = MAX(max_const_size,  const_size);

    // conv3
    conv_option2.out_channels = 256;
    conv_shape(&pvanet.convf, &srpn.conv3,
               &srpn.weight_c3, &srpn.bias_c3,
               &temp_size, &const_size, &conv_option2);
    max_layer_size = MAX(max_layer_size,  flatten_size(&srpn.conv3));
    max_param_size = MAX(max_param_size,  flatten_size(&srpn.weight_c3));
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_const_size = MAX(max_const_size,  const_size);

    // conv5
    conv5x5_option.out_channels = 128;
    conv_shape(&pvanet.convf, &srpn.conv5,
               &srpn.weight_c5, &srpn.bias_c5,
               &temp_size, &const_size, &conv5x5_option);
    max_layer_size = MAX(max_layer_size,  flatten_size(&srpn.conv5));
    max_param_size = MAX(max_param_size,  flatten_size(&srpn.weight_c5));
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_const_size = MAX(max_const_size,  const_size);

    // score1
    conv1x1_option.out_channels = 50;
    conv_shape(&srpn.conv1, &srpn.score1,
               &srpn.weight_s1, &srpn.bias_s1,
               &temp_size, &const_size, &conv1x1_option);
    max_param_size = MAX(max_param_size,  flatten_size(&srpn.weight_s1));
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_const_size = MAX(max_const_size,  const_size);

    // score3
    conv1x1_option.out_channels = 50;
    conv_shape(&srpn.conv3, &srpn.score3,
               &srpn.weight_s3, &srpn.bias_s3,
               &temp_size, &const_size, &conv1x1_option);
    max_param_size = MAX(max_param_size,  flatten_size(&srpn.weight_s3));
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_const_size = MAX(max_const_size,  const_size);

    // score5
    conv1x1_option.out_channels = 50;
    conv_shape(&srpn.conv5, &srpn.score5,
               &srpn.weight_s5, &srpn.bias_s5,
               &temp_size, &const_size, &conv1x1_option);
    max_param_size = MAX(max_param_size,  flatten_size(&srpn.weight_s5));
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_const_size = MAX(max_const_size,  const_size);

    // bbox1
    conv1x1_option.out_channels = 100;
    conv_shape(&srpn.conv1, &srpn.bbox1,
               &srpn.weight_b1, &srpn.bias_b1,
               &temp_size, &const_size, &conv1x1_option);
    max_param_size = MAX(max_param_size,  flatten_size(&srpn.weight_b1));
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_const_size = MAX(max_const_size,  const_size);

    // bbox3
    conv1x1_option.out_channels = 100;
    conv_shape(&srpn.conv3, &srpn.bbox3,
               &srpn.weight_b3, &srpn.bias_b3,
               &temp_size, &const_size, &conv1x1_option);
    max_param_size = MAX(max_param_size,  flatten_size(&srpn.weight_b3));
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_const_size = MAX(max_const_size,  const_size);

    // bbox5
    conv1x1_option.out_channels = 100;
    conv_shape(&srpn.conv5, &srpn.bbox5,
               &srpn.weight_b5, &srpn.bias_b5,
               &temp_size, &const_size, &conv1x1_option);
    max_param_size = MAX(max_param_size,  flatten_size(&srpn.weight_b5));
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_const_size = MAX(max_const_size,  const_size);

    // score
    concat_shape(score_bottoms, &srpn.score, 3);
    max_layer_size = MAX(max_layer_size,  flatten_size(&srpn.score));

    // pred
    srpn.pred.ndim = 3;
    srpn.pred.num_items = srpn.score.num_items;
    for (int n = 0; n < srpn.score.num_items; ++n) {
      srpn.pred.shape[n][0] = 2;
      srpn.pred.shape[n][1] = srpn.score.shape[n][0] / 2;
      srpn.pred.shape[n][2] = srpn.score.shape[n][1];
      srpn.pred.shape[n][3] = srpn.score.shape[n][2];
    }

    // pred reshape
    srpn.pred.ndim = 4;
    srpn.pred.num_items = srpn.score.num_items;
    for (int n = 0; n < srpn.score.num_items; ++n) {
      srpn.pred.shape[n][0] = 2;
      srpn.pred.shape[n][1] = srpn.score.shape[n][0] / 2;
      srpn.pred.shape[n][2] = srpn.score.shape[n][1];
      srpn.pred.shape[n][3] = srpn.score.shape[n][2];
    }

    // bbox
    concat_shape(bbox_bottoms, &srpn.bbox, 3);
    max_layer_size = MAX(max_layer_size,  flatten_size(&srpn.bbox));
    // bbox reshape
    srpn.bbox.ndim = 4;
    for (int n = 0; n < srpn.bbox.num_items; ++n) {
      const int C = srpn.bbox.shape[n][0];
      const int H = srpn.bbox.shape[n][1];
      const int W = srpn.bbox.shape[n][2];
      srpn.bbox.shape[n][0] = C / 4;
      srpn.bbox.shape[n][1] = 4;
      srpn.bbox.shape[n][2] = H;
      srpn.bbox.shape[n][3] = W;
    }

    // img_info
    srpn.img_info.ndim = 1;
    srpn.img_info.num_items = srpn.bbox.num_items;
    for (int n = 0; n < srpn.img_info.num_items; ++n) {
      srpn.img_info.shape[n][0] = 4;
      srpn.img_info.start[n] = n * 4;
    }

    // proposal
    proposal_shape(&srpn.pred, &srpn.roi,
                   &temp_size, &tempint_size, &proposal_option);
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_tempint_size = MAX(max_tempint_size,  tempint_size);
  }

  if (print_network_info) {
    print_tensor_info("rpn_conv1", &srpn.conv1);
    print_tensor_info("rpn_conv3", &srpn.conv3);
    print_tensor_info("rpn_conv5", &srpn.conv5);
    print_tensor_info("rpn_score1", &srpn.score1);
    print_tensor_info("rpn_score3", &srpn.score3);
    print_tensor_info("rpn_score5", &srpn.score5);
    print_tensor_info("rpn_bbox1", &srpn.bbox1);
    print_tensor_info("rpn_bbox3", &srpn.bbox3);
    print_tensor_info("rpn_bbox5", &srpn.bbox5);
    print_tensor_info("rpn_score", &srpn.score);
    print_tensor_info("rpn_pred", &srpn.pred);
    print_tensor_info("rpn_pred_reshape", &srpn.pred);
    print_tensor_info("rpn_bbox", &srpn.bbox);
    print_tensor_info("rpn_bbox_reshape", &srpn.bbox);
    print_tensor_info("img_info", &srpn.img_info);
    print_tensor_info("roi", &srpn.roi);
  }

  // R-CNN
  {
    // roipool
    roipool_shape(&pvanet.convf, &srpn.roi, &rcnn.roipool,
                  &tempint_size, &roipool_option);
    max_layer_size = MAX(max_layer_size, flatten_size(&rcnn.roipool));
    max_tempint_size = MAX(max_tempint_size, tempint_size);

    // roipool reshape
    {
      // calculate total number of RoI-pooled data
      int total_num_rois = 0;
      for (int n = 0; n < rcnn.roipool.num_items; ++n) {
        total_num_rois += rcnn.roipool.shape[n][0];
      }

      // reshape to 2d tensor: total_num_rois x (C * H * W)
      rcnn.roipool_flat.ndim = 2;
      rcnn.roipool_flat.num_items = 1;
      rcnn.roipool_flat.shape[0][0] = total_num_rois;
      rcnn.roipool_flat.shape[0][1] = rcnn.roipool.shape[0][1]
                                      * rcnn.roipool.shape[0][2]
                                      * rcnn.roipool.shape[0][3]; 
    }

    // fc6_1
    fc_option.out_channels = 512;
    fc_option.bias = 0;
    fc_shape(&rcnn.roipool_flat, &rcnn.fc6_1,
             &rcnn.weight6_1, &rcnn.bias6_1,
             &const_size, &fc_option);
    max_layer_size = MAX(max_layer_size,  flatten_size(&rcnn.fc6_1));
    max_param_size = MAX(max_param_size,  flatten_size(&rcnn.weight6_1));
    max_const_size = MAX(max_const_size,  const_size);

    // fc6_2
    fc_option.out_channels = 4096;
    fc_option.bias = 1;
    fc_shape(&rcnn.fc6_1, &rcnn.fc6_2, &rcnn.weight6_2, &rcnn.bias6_2,
             &const_size, &fc_option);
    max_layer_size = MAX(max_layer_size,  flatten_size(&rcnn.fc6_2));
    max_param_size = MAX(max_param_size,  flatten_size(&rcnn.weight6_2));
    max_const_size = MAX(max_const_size,  const_size);

    // fc7_1
    fc_option.out_channels = 4096;
    fc_option.bias = 1;
    fc_shape(&rcnn.fc6_2, &rcnn.fc7_1, &rcnn.weight7_1, &rcnn.bias7_2,
             &const_size, &fc_option);
    max_layer_size = MAX(max_layer_size,  flatten_size(&rcnn.fc7_1));
    max_param_size = MAX(max_param_size,  flatten_size(&rcnn.weight7_1));
    max_const_size = MAX(max_const_size,  const_size);
/*
    // fc7_2
    fc_option.out_channels = 4096;
    fc_option.bias = 1;
    fc_shape(&rcnn.fc7_1, &rcnn.fc7_2, &rcnn.weight7_2, &rcnn.bias7_2,
             &const_size, &fc_option);
    max_layer_size = MAX(max_layer_size,  flatten_size(&rcnn.fc7_2));
    max_param_size = MAX(max_param_size,  flatten_size(&rcnn.weight7_2));
    max_const_size = MAX(max_const_size,  const_size);
*/
    // bbox
    fc_option.out_channels = 84;
    fc_shape(&rcnn.fc7_1, &rcnn.bbox, &rcnn.weight_b, &rcnn.bias_b,
             &const_size, &fc_option);
    max_layer_size = MAX(max_layer_size,  flatten_size(&rcnn.bbox));
    max_param_size = MAX(max_param_size,  flatten_size(&rcnn.weight_b));
    max_const_size = MAX(max_const_size,  const_size);
    // bbox reshape
    rcnn.bbox.ndim = 3;
    rcnn.bbox.num_items = rcnn.roipool.num_items;
    for (int n = 0; n < rcnn.roipool.num_items; ++n) {
      rcnn.bbox.shape[n][0] = rcnn.roipool.shape[n][0];
      rcnn.bbox.shape[n][1] = 21;
      rcnn.bbox.shape[n][2] = 4;
    }

    // score
    fc_option.out_channels = 21;
    fc_shape(&rcnn.fc7_1, &rcnn.score, &rcnn.weight_s, &rcnn.bias_s,
             &const_size, &fc_option);
    max_layer_size = MAX(max_layer_size,  flatten_size(&rcnn.score));
    max_param_size = MAX(max_param_size,  flatten_size(&rcnn.weight_s));
    max_const_size = MAX(max_const_size,  const_size);

    // pred
    rcnn.pred.ndim = 2;
    rcnn.pred.num_items = rcnn.roipool.num_items;
    for (int n = 0; n < rcnn.roipool.num_items; ++n) {
      rcnn.pred.shape[n][0] = rcnn.roipool.shape[n][0];
      rcnn.pred.shape[n][1] = 21;
    }

    // output
    odout_shape(&rcnn.score, &rcnn.out,
                &temp_size, &tempint_size, &odout_option);
    max_layer_size = MAX(max_layer_size,  flatten_size(&rcnn.out));
    max_temp_size = MAX(max_temp_size,  temp_size);
    max_tempint_size = MAX(max_tempint_size,  tempint_size);
  }

  if (print_network_info) {
    print_tensor_info("roipool", &rcnn.roipool);
    print_tensor_info("roipool_flat", &rcnn.roipool_flat);
    print_tensor_info("fc6_1", &rcnn.fc6_1);
    print_tensor_info("fc6_2", &rcnn.fc6_2);
    print_tensor_info("fc7_1", &rcnn.fc7_1);
    print_tensor_info("fc7_2", &rcnn.fc7_2);
    print_tensor_info("bbox", &rcnn.bbox);
    print_tensor_info("score", &rcnn.score);
    print_tensor_info("pred", &rcnn.pred);
    print_tensor_info("out", &rcnn.out);
  }
}

void construct_frcnn_7_1_1(void)
{
  // set fixed options
  {
    conv_option1.num_groups = 1;
    conv_option1.kernel_h = 3;
    conv_option1.kernel_w = 3;
    conv_option1.pad_h = 1;
    conv_option1.pad_w = 1;
    conv_option1.bias = 1;
    conv_option1.stride_h = 2;
    conv_option1.stride_w = 2;

    conv_option2 = conv_option1;
    conv_option2.stride_h = 1;
    conv_option2.stride_w = 1;

    conv1x1_option = conv_option2;
    conv1x1_option.kernel_h = 1;
    conv1x1_option.kernel_w = 1;
    conv1x1_option.pad_h = 0;
    conv1x1_option.pad_w = 0;

    conv5x5_option = conv_option2;
    conv5x5_option.kernel_h = 5;
    conv5x5_option.kernel_w = 5;
    conv5x5_option.pad_h = 2;
    conv5x5_option.pad_w = 2;

    deconv_option = conv_option1;
    deconv_option.num_groups = 512;
    deconv_option.out_channels = 512;
    deconv_option.kernel_h = 4;
    deconv_option.kernel_w = 4;
    deconv_option.bias = 0;

    pool_option.kernel_h = 3;
    pool_option.kernel_w = 3;
    pool_option.stride_h = 2;
    pool_option.stride_w = 2;
    pool_option.pad_h = 0;
    pool_option.pad_w = 0;

    relu_option.negative_slope = 0;

    dropout_option.threshold = 0.5f;
    dropout_option.test = 1;
    dropout_option.scaled = 0;

    proposal_option.scales = &anchor_scales[0];
    proposal_option.ratios = &anchor_ratios[0];
    proposal_option.num_scales = 5;
    proposal_option.num_ratios = 5;
    proposal_option.num_concats = 3;
    proposal_option.base_size = 16;
    proposal_option.feat_stride = 16;
    proposal_option.min_size = 16;
    proposal_option.pre_nms_topn = 6000;
    proposal_option.post_nms_topn = 300;
    proposal_option.nms_thresh = 0.7f;

    odout_option.min_size = 16;
    odout_option.score_thresh = 0.7f;
    odout_option.nms_thresh = 0.3f;

    roipool_option.pooled_height = 6;
    roipool_option.pooled_width = 6;
    roipool_option.spatial_scale = 0.0625;

    fc_option.bias = 1;
  }

  // calculate maximum size
  pvanet.input.num_items = 4;
  pvanet.input.ndim = 3;
  for (int n = 0; n < pvanet.input.num_items; ++n) {
    pvanet.input.shape[n][0] = 3;
    pvanet.input.shape[n][1] = 640;
    pvanet.input.shape[n][2] = 1024;
    pvanet.input.start[n] = n * 3 * 640 * 1024;
  }
  shape_frcnn_7_1_1(1);

  // memory allocation
  {
    // total memory size (in byte) always allocated in main memory
    long int space_cpu = 0;
    // allocated size in GPU memory (GPU mode) or main memory (CPU mode)
    long int space = 0;

    printf("Max layer size = %d\n", max_layer_size);
    printf("Max param size = %d\n", max_param_size);
    printf("Max temp size = %d\n", max_temp_size);
    printf("Max tempint size = %d\n", max_tempint_size);
    printf("Max const size = %d\n", max_const_size);

    // space for data loading
    {
      input_data = (real*)malloc(flatten_size(&pvanet.input) * sizeof(real));
      output_data = (real*)malloc(max_layer_size * sizeof(real));
      true_data = (real*)malloc(max_layer_size * sizeof(real));
      param_data = (real*)malloc(max_param_size * sizeof(real));
      space_cpu += sizeof(real) * (flatten_size(&pvanet.input)
                                   + max_layer_size * 2 + max_param_size);
    }

    // space required for forward-pass
    {
      const int num_anchors = proposal_option.num_scales
                              * proposal_option.num_ratios
                              * proposal_option.num_concats;
    #ifdef GPU
      cudaMalloc(&layer1_data, max_layer_size * sizeof(real));
      cudaMalloc(&layer2_data, max_layer_size * sizeof(real));
      cudaMalloc(&layer3_data, max_layer_size * sizeof(real));
      cudaMalloc(&temp_data, max_temp_size * sizeof(real));
      cudaMalloc(&const_data, max_const_size * sizeof(real));
      cudaMalloc(&tempint_data, max_tempint_size * sizeof(int));
      cudaMalloc(&backup1_data, max_layer_size * sizeof(real));
      cudaMalloc(&backup2_data, max_layer_size * sizeof(real));
      cudaMalloc(&anchors, num_anchors * 4 * sizeof(real));
    #else
      layer1_data = (real*)malloc(max_layer_size * sizeof(real));
      layer2_data = (real*)malloc(max_layer_size * sizeof(real));
      layer3_data = (real*)malloc(max_layer_size * sizeof(real));
      temp_data = (real*)malloc(max_temp_size * sizeof(real));
      const_data = (real*)malloc(max_const_size * sizeof(real));
      tempint_data = (int*)malloc(max_tempint_size * sizeof(int));
      backup1_data = (real*)malloc(max_layer_size * sizeof(real));
      backup2_data = (real*)malloc(max_layer_size * sizeof(real));
      anchors = (real*)malloc(num_anchors * 4 * sizeof(real));
    #endif
      space += sizeof(real) * (max_layer_size * 5 + max_temp_size
                               + max_const_size + num_anchors * 4)
             + sizeof(int) * (max_tempint_size);

      proposal_temp = (real*)malloc(max_temp_size * sizeof(real));
      proposal_tempint = (int*)malloc(max_tempint_size * sizeof(int));
      space_cpu += sizeof(real) * (max_temp_size)
                   + sizeof(int) * (max_tempint_size);
    }

    // PVANET parameters
    {
      space += malloc_tensor(&pvanet.weight1_1);
      space += malloc_tensor(&pvanet.bias1_1);
      space += malloc_tensor(&pvanet.weight1_2);
      space += malloc_tensor(&pvanet.bias1_2);
      space += malloc_tensor(&pvanet.weight2_1);
      space += malloc_tensor(&pvanet.bias2_1);
      space += malloc_tensor(&pvanet.weight2_2);
      space += malloc_tensor(&pvanet.bias2_2);
      space += malloc_tensor(&pvanet.weight3_1);
      space += malloc_tensor(&pvanet.bias3_1);
      space += malloc_tensor(&pvanet.weight3_2);
      space += malloc_tensor(&pvanet.bias3_2);
      space += malloc_tensor(&pvanet.weight3_3);
      space += malloc_tensor(&pvanet.bias3_3);
      space += malloc_tensor(&pvanet.weight4_1);
      space += malloc_tensor(&pvanet.bias4_1);
      space += malloc_tensor(&pvanet.weight4_2);
      space += malloc_tensor(&pvanet.bias4_2);
      space += malloc_tensor(&pvanet.weight4_3);
      space += malloc_tensor(&pvanet.bias4_3);
      space += malloc_tensor(&pvanet.weight5_1);
      space += malloc_tensor(&pvanet.bias5_1);
      space += malloc_tensor(&pvanet.weight5_2);
      space += malloc_tensor(&pvanet.bias5_2);
      space += malloc_tensor(&pvanet.weight5_3);
      space += malloc_tensor(&pvanet.bias5_3);
      space += malloc_tensor(&pvanet.weight_up);
      space += malloc_tensor(&pvanet.weightf);
      space += malloc_tensor(&pvanet.biasf);
    }

    // SRPN parameters & layers
    {
      space += malloc_tensor(&srpn.weight_c1);
      space += malloc_tensor(&srpn.bias_c1);
      space += malloc_tensor(&srpn.weight_c3);
      space += malloc_tensor(&srpn.bias_c3);
      space += malloc_tensor(&srpn.weight_c5);
      space += malloc_tensor(&srpn.bias_c5);
      space += malloc_tensor(&srpn.weight_s1);
      space += malloc_tensor(&srpn.bias_s1);
      space += malloc_tensor(&srpn.weight_s3);
      space += malloc_tensor(&srpn.bias_s3);
      space += malloc_tensor(&srpn.weight_s5);
      space += malloc_tensor(&srpn.bias_s5);
      space += malloc_tensor(&srpn.weight_b1);
      space += malloc_tensor(&srpn.bias_b1);
      space += malloc_tensor(&srpn.weight_b3);
      space += malloc_tensor(&srpn.bias_b3);
      space += malloc_tensor(&srpn.weight_b5);
      space += malloc_tensor(&srpn.bias_b5);

      space += malloc_tensor(&srpn.score1);
      space += malloc_tensor(&srpn.score3);
      space += malloc_tensor(&srpn.score5);
      space += malloc_tensor(&srpn.bbox1);
      space += malloc_tensor(&srpn.bbox3);
      space += malloc_tensor(&srpn.bbox5);
      space += malloc_tensor(&srpn.roi);

      // for convenience, img_info.data is always allocated in main memory
      srpn.img_info.data
          = (real*)malloc(flatten_size(&srpn.img_info) * sizeof(real));
      space_cpu += sizeof(real) * flatten_size(&srpn.img_info);
    }

    // RCNN parameters
    {
      space += malloc_tensor(&rcnn.weight6_1);
      space += malloc_tensor(&rcnn.weight6_2);
      space += malloc_tensor(&rcnn.bias6_2);
      space += malloc_tensor(&rcnn.weight7_1);
      //space += malloc_tensor(&rcnn.weight7_2);
      space += malloc_tensor(&rcnn.bias7_2);
      space += malloc_tensor(&rcnn.weight_s);
      space += malloc_tensor(&rcnn.bias_s);
      space += malloc_tensor(&rcnn.weight_b);
      space += malloc_tensor(&rcnn.bias_b);
    }

    // print total memory size required
    {
    #ifdef GPU
      printf("%ldMB of main memory allocated\n",
             DIV_THEN_CEIL(space_cpu,  1000000));
      printf("%ldMB of GPU memory allocated\n",
             DIV_THEN_CEIL(space,  1000000));
    #else
      printf("%ldMB of main memory allocated\n",
             DIV_THEN_CEIL(space_cpu + space,  1000000));
    #endif
    }
  }

  // data initialization
  {
  #ifdef GPU
    for (int i = 0; i < max_const_size; ++i) {
      output_data[i] = 1;
    }
    cudaMemcpy(const_data, output_data, max_const_size * sizeof(real),
               cudaMemcpyHostToDevice);
  #else
    for (int i = 0; i < max_const_size; ++i) {
      const_data[i] = 1;
    }
  #endif
  }

  // anchor generation for proposal layer
  {
  #ifdef GPU
    const int num_anchors = proposal_option.num_scales
                            * proposal_option.num_ratios
                            * proposal_option.num_concats;
    generate_anchors(param_data, &proposal_option);
    cudaMemcpy(anchors, param_data, num_anchors * 4 * sizeof(real),
               cudaMemcpyHostToDevice);
  #else
    generate_anchors(anchors, &proposal_option);
  #endif
  }

  // PVANET parameter loading
  {
    load_tensor("../data/temp/conv1_1_param0.bin", &pvanet.weight1_1, param_data);
    load_tensor("../data/temp/conv1_1_param1.bin", &pvanet.bias1_1, param_data);
    load_tensor("../data/temp/conv1_2_param0.bin", &pvanet.weight1_2, param_data);
    load_tensor("../data/temp/conv1_2_param1.bin", &pvanet.bias1_2, param_data);
    load_tensor("../data/temp/conv2_1_param0.bin", &pvanet.weight2_1, param_data);
    load_tensor("../data/temp/conv2_1_param1.bin", &pvanet.bias2_1, param_data);
    load_tensor("../data/temp/conv2_2_param0.bin", &pvanet.weight2_2, param_data);
    load_tensor("../data/temp/conv2_2_param1.bin", &pvanet.bias2_2, param_data);
    load_tensor("../data/temp/conv3_1_param0.bin", &pvanet.weight3_1, param_data);
    load_tensor("../data/temp/conv3_1_param1.bin", &pvanet.bias3_1, param_data);
    load_tensor("../data/temp/conv3_2_param0.bin", &pvanet.weight3_2, param_data);
    load_tensor("../data/temp/conv3_2_param1.bin", &pvanet.bias3_2, param_data);
    load_tensor("../data/temp/conv3_3_param0.bin", &pvanet.weight3_3, param_data);
    load_tensor("../data/temp/conv3_3_param1.bin", &pvanet.bias3_3, param_data);
    load_tensor("../data/temp/conv4_1_param0.bin", &pvanet.weight4_1, param_data);
    load_tensor("../data/temp/conv4_1_param1.bin", &pvanet.bias4_1, param_data);
    load_tensor("../data/temp/conv4_2_param0.bin", &pvanet.weight4_2, param_data);
    load_tensor("../data/temp/conv4_2_param1.bin", &pvanet.bias4_2, param_data);
    load_tensor("../data/temp/conv4_3_param0.bin", &pvanet.weight4_3, param_data);
    load_tensor("../data/temp/conv4_3_param1.bin", &pvanet.bias4_3, param_data);
    load_tensor("../data/temp/conv5_1_param0.bin", &pvanet.weight5_1, param_data);
    load_tensor("../data/temp/conv5_1_param1.bin", &pvanet.bias5_1, param_data);
    load_tensor("../data/temp/conv5_2_param0.bin", &pvanet.weight5_2, param_data);
    load_tensor("../data/temp/conv5_2_param1.bin", &pvanet.bias5_2, param_data);
    load_tensor("../data/temp/conv5_3_param0.bin", &pvanet.weight5_3, param_data);
    load_tensor("../data/temp/conv5_3_param1.bin", &pvanet.bias5_3, param_data);
    load_tensor("../data/temp/upsample_param0.bin", &pvanet.weight_up, param_data);
    load_tensor("../data/temp/convf_param0.bin", &pvanet.weightf, param_data);
    load_tensor("../data/temp/convf_param1.bin", &pvanet.biasf, param_data);
  }

  // SRPN parameter loading
  {
    load_tensor("../data/temp/rpn_conv1_param0.bin", &srpn.weight_c1, param_data);
    load_tensor("../data/temp/rpn_conv1_param1.bin", &srpn.bias_c1, param_data);
    load_tensor("../data/temp/rpn_conv3_param0.bin", &srpn.weight_c3, param_data);
    load_tensor("../data/temp/rpn_conv3_param1.bin", &srpn.bias_c3, param_data);
    load_tensor("../data/temp/rpn_conv5_param0.bin", &srpn.weight_c5, param_data);
    load_tensor("../data/temp/rpn_conv5_param1.bin", &srpn.bias_c5, param_data);
    load_tensor("../data/temp/rpn_cls_score1_param0.bin", &srpn.weight_s1, param_data);
    load_tensor("../data/temp/rpn_cls_score1_param1.bin", &srpn.bias_s1, param_data);
    load_tensor("../data/temp/rpn_cls_score3_param0.bin", &srpn.weight_s3, param_data);
    load_tensor("../data/temp/rpn_cls_score3_param1.bin", &srpn.bias_s3, param_data);
    load_tensor("../data/temp/rpn_cls_score5_param0.bin", &srpn.weight_s5, param_data);
    load_tensor("../data/temp/rpn_cls_score5_param1.bin", &srpn.bias_s5, param_data);
    load_tensor("../data/temp/rpn_bbox_pred1_param0.bin", &srpn.weight_b1, param_data);
    load_tensor("../data/temp/rpn_bbox_pred1_param1.bin", &srpn.bias_b1, param_data);
    load_tensor("../data/temp/rpn_bbox_pred3_param0.bin", &srpn.weight_b3, param_data);
    load_tensor("../data/temp/rpn_bbox_pred3_param1.bin", &srpn.bias_b3, param_data);
    load_tensor("../data/temp/rpn_bbox_pred5_param0.bin", &srpn.weight_b5, param_data);
    load_tensor("../data/temp/rpn_bbox_pred5_param1.bin", &srpn.bias_b5, param_data);
  }

  // RCNN parameter loading
  {
    load_tensor("../data/temp/fc6_1_param0.bin", &rcnn.weight6_1, param_data);
    load_tensor("../data/temp/fc6_2_param0.bin", &rcnn.weight6_2, param_data);
    load_tensor("../data/temp/fc6_param1.bin", &rcnn.bias6_2, param_data);
    load_tensor("../data/temp/fc7_param0.bin", &rcnn.weight7_1, param_data);
    //load_tensor("../data/temp/fc7_2_param0.bin", &rcnn.weight7_2, param_data);
    load_tensor("../data/temp/fc7_param1.bin", &rcnn.bias7_2, param_data);
    load_tensor("../data/temp/cls_score_param0.bin", &rcnn.weight_s, param_data);
    load_tensor("../data/temp/cls_score_param1.bin", &rcnn.bias_s, param_data);
    load_tensor("../data/temp/bbox_pred_param0.bin", &rcnn.weight_b, param_data);
    load_tensor("../data/temp/bbox_pred_param1.bin", &rcnn.bias_b, param_data);
  }

  // acquire CuBLAS handle
  #ifdef GPU
  {
    conv_option1.handle = (cublasHandle_t*)malloc(sizeof(cublasHandle_t));
    if (cublasCreate((cublasHandle_t*)conv_option1.handle)
          != CUBLAS_STATUS_SUCCESS) {
      printf("cublas creation failed\n");
    }
    conv_option2.handle = conv_option1.handle;
    conv1x1_option.handle = conv_option1.handle;
    conv5x5_option.handle = conv_option1.handle;
    deconv_option.handle = conv_option1.handle;
    fc_option.handle = conv_option1.handle;
  }
  #endif
}

void frcnn_memset(void)
{
  memset(input_data, 0, flatten_size(&pvanet.input) * sizeof(real));
  memset(output_data, 0, max_layer_size * sizeof(real));
  memset(true_data, 0, max_layer_size * sizeof(real));
  memset(param_data, 0, max_param_size * sizeof(real));

  #ifdef GPU
  cudaMemset(layer1_data, 0, max_layer_size * sizeof(real));
  cudaMemset(layer2_data, 0, max_layer_size * sizeof(real));
  cudaMemset(layer3_data, 0, max_layer_size * sizeof(real));
  cudaMemset(temp_data, 0, max_temp_size * sizeof(real));
  cudaMemset(const_data, 0, max_const_size * sizeof(real));
  cudaMemset(tempint_data, 0, max_tempint_size * sizeof(int));
  cudaMemset(backup1_data, 0, max_layer_size * sizeof(real));
  cudaMemset(backup2_data, 0, max_layer_size * sizeof(real));
  #else
  memset(layer1_data, 0, max_layer_size * sizeof(real));
  memset(layer2_data, 0, max_layer_size * sizeof(real));
  memset(layer3_data, 0, max_layer_size * sizeof(real));
  memset(temp_data, 0, max_temp_size * sizeof(real));
  memset(const_data, 0, max_const_size * sizeof(real));
  memset(tempint_data, 0, max_tempint_size * sizeof(int));
  memset(backup1_data, 0, max_layer_size * sizeof(real));
  memset(backup2_data, 0, max_layer_size * sizeof(real));
  #endif

  memset(proposal_temp, 0, max_temp_size * sizeof(real));
  memset(proposal_tempint, 0, max_tempint_size * sizeof(int));

  #ifdef GPU
  cudaMemset(srpn.score1.data, 0, flatten_size(&srpn.score1) * sizeof(real));
  cudaMemset(srpn.score3.data, 0, flatten_size(&srpn.score3) * sizeof(real));
  cudaMemset(srpn.score5.data, 0, flatten_size(&srpn.score5) * sizeof(real));
  cudaMemset(srpn.bbox1.data, 0, flatten_size(&srpn.bbox1) * sizeof(real));
  cudaMemset(srpn.bbox3.data, 0, flatten_size(&srpn.bbox3) * sizeof(real));
  cudaMemset(srpn.bbox5.data, 0, flatten_size(&srpn.bbox5) * sizeof(real));
  cudaMemset(srpn.roi.data, 0, flatten_size(&srpn.roi) * sizeof(real));
  #else
  memset(srpn.score1.data, 0, flatten_size(&srpn.score1) * sizeof(real));
  memset(srpn.score3.data, 0, flatten_size(&srpn.score3) * sizeof(real));
  memset(srpn.score5.data, 0, flatten_size(&srpn.score5) * sizeof(real));
  memset(srpn.bbox1.data, 0, flatten_size(&srpn.bbox1) * sizeof(real));
  memset(srpn.bbox3.data, 0, flatten_size(&srpn.bbox3) * sizeof(real));
  memset(srpn.bbox5.data, 0, flatten_size(&srpn.bbox5) * sizeof(real));
  memset(srpn.roi.data, 0, flatten_size(&srpn.roi) * sizeof(real));
  #endif

  memset(srpn.img_info.data, 0, flatten_size(&srpn.img_info) * sizeof(real));
}

void destruct_frcnn_7_1_1(void)
{
  // space always allocated in CPU memory
  {
    if (true_data) free(true_data);
    if (input_data) free(input_data);
    if (output_data) free(output_data);
    if (param_data) free(param_data);

    if (proposal_temp) free(proposal_temp);
    if (proposal_tempint) free(proposal_tempint);

    if (srpn.img_info.data) free(srpn.img_info.data);
  }

  #ifdef GPU
  // space allocated in GPU memory
  {
    if (layer1_data) cudaFree(layer1_data);
    if (layer2_data) cudaFree(layer2_data);
    if (layer3_data) cudaFree(layer3_data);
    if (backup1_data) cudaFree(backup1_data);
    if (backup2_data) cudaFree(backup2_data);
    if (temp_data) cudaFree(temp_data);
    if (tempint_data) cudaFree(tempint_data);
    if (const_data) cudaFree(const_data);
    if (anchors) cudaFree(anchors);

    // PVANET parameters
    cudaFree(pvanet.weight1_1.data);
    cudaFree(pvanet.bias1_1.data);
    cudaFree(pvanet.weight1_2.data);
    cudaFree(pvanet.bias1_2.data);
    cudaFree(pvanet.weight2_1.data);
    cudaFree(pvanet.bias2_1.data);
    cudaFree(pvanet.weight2_2.data);
    cudaFree(pvanet.bias2_2.data);
    cudaFree(pvanet.weight3_1.data);
    cudaFree(pvanet.bias3_1.data);
    cudaFree(pvanet.weight3_2.data);
    cudaFree(pvanet.bias3_2.data);
    cudaFree(pvanet.weight3_3.data);
    cudaFree(pvanet.bias3_3.data);
    cudaFree(pvanet.weight4_1.data);
    cudaFree(pvanet.bias4_1.data);
    cudaFree(pvanet.weight4_2.data);
    cudaFree(pvanet.bias4_2.data);
    cudaFree(pvanet.weight4_3.data);
    cudaFree(pvanet.bias4_3.data);
    cudaFree(pvanet.weight5_1.data);
    cudaFree(pvanet.bias5_1.data);
    cudaFree(pvanet.weight5_2.data);
    cudaFree(pvanet.bias5_2.data);
    cudaFree(pvanet.weight5_3.data);
    cudaFree(pvanet.bias5_3.data);
    cudaFree(pvanet.weight_up.data);
    cudaFree(pvanet.weightf.data);
    cudaFree(pvanet.biasf.data);

    // SRPN parameters
    cudaFree(srpn.weight_c1.data);
    cudaFree(srpn.bias_c1.data);
    cudaFree(srpn.weight_c3.data);
    cudaFree(srpn.bias_c3.data);
    cudaFree(srpn.weight_c5.data);
    cudaFree(srpn.bias_c5.data);
    cudaFree(srpn.weight_s1.data);
    cudaFree(srpn.bias_s1.data);
    cudaFree(srpn.weight_s3.data);
    cudaFree(srpn.bias_s3.data);
    cudaFree(srpn.weight_s5.data);
    cudaFree(srpn.bias_s5.data);
    cudaFree(srpn.weight_b1.data);
    cudaFree(srpn.bias_b1.data);
    cudaFree(srpn.weight_b3.data);
    cudaFree(srpn.bias_b3.data);
    cudaFree(srpn.weight_b5.data);
    cudaFree(srpn.bias_b5.data);

    // SRPN layers
    cudaFree(srpn.score1.data);
    cudaFree(srpn.score3.data);
    cudaFree(srpn.score5.data);
    cudaFree(srpn.bbox1.data);
    cudaFree(srpn.bbox3.data);
    cudaFree(srpn.bbox5.data);
    cudaFree(srpn.roi.data);

    // RCNN parameters
    cudaFree(rcnn.weight6_1.data);
    cudaFree(rcnn.weight6_2.data);
    cudaFree(rcnn.bias6_2.data);
    cudaFree(rcnn.weight7_1.data);
    //cudaFree(rcnn.weight7_2.data);
    cudaFree(rcnn.bias7_2.data);
    cudaFree(rcnn.weight_s.data);
    cudaFree(rcnn.bias_s.data);
    cudaFree(rcnn.weight_b.data);
    cudaFree(rcnn.bias_b.data);

    // release CuBLAS handle
    if (cublasDestroy(*((cublasHandle_t*)conv_option1.handle))
        != CUBLAS_STATUS_SUCCESS) {
      printf("cublas destruction failed\n");
    }
    free(conv_option1.handle);
  }

  #else
  // space allocated in CPU memory
  {
    if (layer1_data) free(layer1_data);
    if (layer2_data) free(layer2_data);
    if (layer3_data) free(layer3_data);
    if (backup1_data) free(backup1_data);
    if (backup2_data) free(backup2_data);
    if (temp_data) free(temp_data);
    if (tempint_data) free(tempint_data);
    if (const_data) free(const_data);
    if (anchors) free(anchors);

    // PVANET parameters
    free(pvanet.weight1_1.data);
    free(pvanet.bias1_1.data);
    free(pvanet.weight1_2.data);
    free(pvanet.bias1_2.data);
    free(pvanet.weight2_1.data);
    free(pvanet.bias2_1.data);
    free(pvanet.weight2_2.data);
    free(pvanet.bias2_2.data);
    free(pvanet.weight3_1.data);
    free(pvanet.bias3_1.data);
    free(pvanet.weight3_2.data);
    free(pvanet.bias3_2.data);
    free(pvanet.weight3_3.data);
    free(pvanet.bias3_3.data);
    free(pvanet.weight4_1.data);
    free(pvanet.bias4_1.data);
    free(pvanet.weight4_2.data);
    free(pvanet.bias4_2.data);
    free(pvanet.weight4_3.data);
    free(pvanet.bias4_3.data);
    free(pvanet.weight5_1.data);
    free(pvanet.bias5_1.data);
    free(pvanet.weight5_2.data);
    free(pvanet.bias5_2.data);
    free(pvanet.weight5_3.data);
    free(pvanet.bias5_3.data);
    free(pvanet.weight_up.data);
    free(pvanet.weightf.data);
    free(pvanet.biasf.data);

    // SRPN parameters
    free(srpn.weight_c1.data);
    free(srpn.bias_c1.data);
    free(srpn.weight_c3.data);
    free(srpn.bias_c3.data);
    free(srpn.weight_c5.data);
    free(srpn.bias_c5.data);
    free(srpn.weight_s1.data);
    free(srpn.bias_s1.data);
    free(srpn.weight_s3.data);
    free(srpn.bias_s3.data);
    free(srpn.weight_s5.data);
    free(srpn.bias_s5.data);
    free(srpn.weight_b1.data);
    free(srpn.bias_b1.data);
    free(srpn.weight_b3.data);
    free(srpn.bias_b3.data);
    free(srpn.weight_b5.data);
    free(srpn.bias_b5.data);

    // SRPN layers
    free(srpn.score1.data);
    free(srpn.score3.data);
    free(srpn.score5.data);
    free(srpn.bbox1.data);
    free(srpn.bbox3.data);
    free(srpn.bbox5.data);
    free(srpn.roi.data);

    // RCNN parameters
    free(rcnn.weight6_1.data);
    free(rcnn.weight6_2.data);
    free(rcnn.bias6_2.data);
    free(rcnn.weight7_1.data);
    //free(rcnn.weight7_2.data);
    free(rcnn.bias7_2.data);
    free(rcnn.weight_s.data);
    free(rcnn.bias_s.data);
    free(rcnn.weight_b.data);
    free(rcnn.bias_b.data);
  }
  #endif
}

void prepare_input(const char* const filename[], const int num_images)
{
  pvanet.input.data = input_data;
  pvanet.input.ndim = 3;
  pvanet.input.num_items = 0;
  pvanet.input.start[0] = 0;
  srpn.img_info.ndim = 1;
  srpn.img_info.num_items = 0;

  for (int i = 0; i < num_images; ++i) {
    load_image(filename[i], &pvanet.input, &srpn.img_info);
  }

  #ifdef GPU
  cudaMemcpyAsync(layer1_data, input_data,
                  flatten_size(&pvanet.input) * sizeof(real),
                  cudaMemcpyHostToDevice);
  #else
  memcpy(layer1_data, input_data,
         flatten_size(&pvanet.input) * sizeof(real));
  #endif

  // network reshape
  shape_frcnn_7_1_1(0);

  print_tensor_info("input data loaded", &pvanet.input);
}

void get_output(const int image_start_index, FILE* fp)
{
  // retrieve output
  {
    const int output_size = flatten_size(&rcnn.out);

  #ifdef GPU
    cudaMemcpyAsync(output_data, rcnn.out.data,
                    output_size * sizeof(real),
                    cudaMemcpyDeviceToHost);
  #else
    memcpy(output_data, rcnn.out.data, output_size * sizeof(real));
  #endif
  }

  // print output
  {
    for (int n = 0; n < rcnn.out.num_items; ++n) {
      const int image_index = image_start_index + n;
      const real* const p_out_item = output_data + rcnn.out.start[n];
      for (int i = 0; i < rcnn.out.shape[n][0]; ++i) {
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
  construct_frcnn_7_1_1();

  // load a text file containing image filenames to be tested
  {
    char line1[1024], line2[1024], line3[1024], line4[1024];
    char* line[] = { line1, line2, line3, line4 };
    int total_count = 0, count = 0;
    FILE* fp_list = fopen(argv[1], "r");
    FILE* fp_out = fopen(argv[2], "wb");

    if (!fp_list) {
      printf("File not found: %s\n", argv[1]);
    }
    if (!fp_out) {
      printf("File write error: %s\n", argv[2]);
    }

    while (fgets(line[count], 1024, fp_list)) {
      const int len = strlen(line[count]);
      line[count][len - 1] = 0;
      ++count;
      if (count == 4) {
        // input data loading
        prepare_input((const char * const *)&line, count);

        // forward-pass
        forward_frcnn_7_1_1();

        // retrieve output & save to file
        get_output(total_count, fp_out);

        total_count += count;
        count = 0;
      }
    }

    if (count > 0) {
      prepare_input((const char * const *)&line, count);
      printf("forward-pass start\n");
      forward_frcnn_7_1_1();
      printf("forward-pass end\n");
      get_output(total_count, fp_out);
    }

    fclose(fp_list);
    fclose(fp_out);
  }


  // end
  destruct_frcnn_7_1_1();

  return 0;
}
