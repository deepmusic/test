#include "nets/net_factory.h"

void setup_faster_rcnn(Net* const net,
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
    int rpn_score_shape[] = { 2, -1, 0 };
    add_reshape_layer(net, "rpn_cls_score_reshape",
                      "rpn_cls_score", "rpn_cls_score", rpn_score_shape, 3);
  }

  add_softmax_layer(net, "rpn_cls_pred", "rpn_cls_score", "rpn_cls_score",
                    0);

  {
    int rpn_pred_shape[] = { 50, -1, 0 };
    add_reshape_layer(net, "rpn_cls_pred_reshape",
                      "rpn_cls_score", "rpn_cls_score", rpn_pred_shape, 3);
  }

  add_conv_layer(net, "rpn_bbox_pred", "rpn_conv1", "rpn_bbox_pred",
                 NULL, NULL,
                 1, 100, 1, 1, 1, 1, 0, 0, 1);

  {
    real anchor_scales[5] = { 3.0f, 6.0f, 9.0f, 16.0f, 32.0f };
    real anchor_ratios[5] = { 0.5f, 0.667f, 1.0f, 1.5f, 2.0f };
    add_proposal_layer(net, "proposal", "rpn_cls_score", "rpn_bbox_pred",
                       "im_info", "rpn_roi",
                       anchor_scales, 5, anchor_ratios, 5, 16, 16, 16,
                       pre_nms_topn, post_nms_topn, 0.7f, 0, 0.7f);
  }

  add_roipool_layer(net, "rcnn_roipool", rcnn_input_name,
                    "rpn_roi", "rcnn_roipool", 6, 6, 0.062500f, 1);
  get_tensor_by_name(net, "rcnn_roipool")->data_type = PRIVATE_DATA;

  if (fc_compress) {
    add_fc_layer(net, "fc6_L", "rcnn_roipool", "fc6_L", NULL, NULL,
                 fc6_dim, 0);
    add_fc_layer(net, "fc6_U", "fc6_L", "fc6_U", NULL, NULL, 4096, 1);
    add_relu_layer(net, "relu6", "fc6_U", "fc6_U", 0);
    add_dropout_layer(net, "drop6", "fc6_U", "fc6_U", 0, 1, 1);

    add_fc_layer(net, "fc7_L", "fc6_U", "fc7_L", NULL, NULL, fc7_dim, 0);
    add_fc_layer(net, "fc7_U", "fc7_L", "fc7_U", NULL, NULL, 4096, 1);
    add_relu_layer(net, "relu7", "fc7_U", "fc7_U", 0);
    add_dropout_layer(net, "drop7", "fc7_U", "fc7_U", 0, 1, 1);

    add_fc_layer(net, "cls_score", "fc7_U", "cls_score", NULL, NULL, 21, 1);
    add_fc_layer(net, "bbox_pred", "fc7_U", "bbox_pred", NULL, NULL, 84, 1);
  }
  else {
    add_fc_layer(net, "fc6", "rcnn_roipool", "fc6", NULL, NULL, 4096, 1);
    add_relu_layer(net, "relu6", "fc6", "fc6", 0);
    add_dropout_layer(net, "drop6", "fc6", "fc6", 0, 1, 1);

    add_fc_layer(net, "fc7", "fc6", "fc7", NULL, NULL, 4096, 1);
    add_relu_layer(net, "relu7", "fc7", "fc7", 0);
    add_dropout_layer(net, "drop7", "fc7", "fc7", 0, 1, 1);

    add_fc_layer(net, "cls_score", "fc7", "cls_score", NULL, NULL, 21, 1);
    add_fc_layer(net, "bbox_pred", "fc7", "bbox_pred", NULL, NULL, 84, 1);
  }

  add_softmax_layer(net, "cls_pred", "cls_score", "cls_score", 1);

  add_odout_layer(net, "out", "cls_score", "bbox_pred", "rpn_roi",
                  "im_info", "out", 16, post_nms_topn, 0.7f, 0.4f, 0, 0.5f);
}
