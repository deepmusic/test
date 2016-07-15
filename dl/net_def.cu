#include "layer.h"
void setup_shared_conv_sub(Net* const net)
{
  add_data_layer(net, "input-data", "data", "img_info");
  add_conv_layer(net, "conv1_1_conv", "data", "conv1_1_conv", NULL, NULL, 1, 16, 7, 7, 2, 2, 3, 3, 1, 0);
  add_scale_const_layer(net, "conv1_1_neg", "conv1_1_conv", "conv1_1_neg", -1.000000, 0.000000, 1);
  { const char* const names[] = { "conv1_1_conv", "conv1_1_neg" }; add_concat_layer(net, "conv1_1_concat", names, "conv1_1", 2); }
  add_scale_channel_layer(net, "conv1_1_scale", "conv1_1", "conv1_1", NULL, NULL, 1);
  add_relu_layer(net, "conv1_1_relu", "conv1_1", "conv1_1", 0);
  add_pool_layer(net, "pool1", "conv1_1", "pool1", 3, 3, 2, 2, 0, 0);
  add_conv_layer(net, "conv2_1_1_conv", "pool1", "conv2_1_1", NULL, NULL, 1, 24, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv2_1_1_relu", "conv2_1_1", "conv2_1_1", 0);
  add_conv_layer(net, "conv2_1_2_conv", "conv2_1_1", "conv2_1_2_conv", NULL, NULL, 1, 24, 3, 3, 1, 1, 1, 1, 1, 0);
  add_scale_const_layer(net, "conv2_1_2_neg", "conv2_1_2_conv", "conv2_1_2_neg", -1.000000, 0.000000, 1);
  { const char* const names[] = { "conv2_1_2_conv", "conv2_1_2_neg" }; add_concat_layer(net, "conv2_1_2_concat", names, "conv2_1_2", 2); }
  add_scale_channel_layer(net, "conv2_1_2_scale", "conv2_1_2", "conv2_1_2", NULL, NULL, 1);
  add_relu_layer(net, "conv2_1_2_relu", "conv2_1_2", "conv2_1_2", 0);
  add_conv_layer(net, "conv2_1_3_conv", "conv2_1_2", "conv2_1_3", NULL, NULL, 1, 64, 1, 1, 1, 1, 0, 0, 1, 0);
  add_conv_layer(net, "conv2_1_proj", "pool1", "conv2_1_proj", NULL, NULL, 1, 64, 1, 1, 1, 1, 0, 0, 1, 0);
  { const char* const names[] = { "conv2_1_3", "conv2_1_proj" }; add_eltwise_layer(net, "conv2_1", names, "conv2_1", 2); }
  add_conv_layer(net, "conv2_2_1_conv", "conv2_1", "conv2_2_1", NULL, NULL, 1, 24, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv2_2_1_relu", "conv2_2_1", "conv2_2_1", 0);
  add_conv_layer(net, "conv2_2_2_conv", "conv2_2_1", "conv2_2_2_conv", NULL, NULL, 1, 24, 3, 3, 1, 1, 1, 1, 1, 0);
  add_scale_const_layer(net, "conv2_2_2_neg", "conv2_2_2_conv", "conv2_2_2_neg", -1.000000, 0.000000, 1);
  { const char* const names[] = { "conv2_2_2_conv", "conv2_2_2_neg" }; add_concat_layer(net, "conv2_2_2_concat", names, "conv2_2_2", 2); }
  add_scale_channel_layer(net, "conv2_2_2_scale", "conv2_2_2", "conv2_2_2", NULL, NULL, 1);
  add_relu_layer(net, "conv2_2_2_relu", "conv2_2_2", "conv2_2_2", 0);
  add_conv_layer(net, "conv2_2_3_conv", "conv2_2_2", "conv2_2_3", NULL, NULL, 1, 64, 1, 1, 1, 1, 0, 0, 1, 0);
  { const char* const names[] = { "conv2_2_3", "conv2_1" }; add_eltwise_layer(net, "conv2_2", names, "conv2_2", 2); }
  add_conv_layer(net, "conv2_3_1_conv", "conv2_2", "conv2_3_1", NULL, NULL, 1, 24, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv2_3_1_relu", "conv2_3_1", "conv2_3_1", 0);
  add_conv_layer(net, "conv2_3_2_conv", "conv2_3_1", "conv2_3_2_conv", NULL, NULL, 1, 24, 3, 3, 1, 1, 1, 1, 1, 0);
  add_scale_const_layer(net, "conv2_3_2_neg", "conv2_3_2_conv", "conv2_3_2_neg", -1.000000, 0.000000, 1);
  { const char* const names[] = { "conv2_3_2_conv", "conv2_3_2_neg" }; add_concat_layer(net, "conv2_3_2_concat", names, "conv2_3_2", 2); }
  add_scale_channel_layer(net, "conv2_3_2_scale", "conv2_3_2", "conv2_3_2", NULL, NULL, 1);
  add_relu_layer(net, "conv2_3_2_relu", "conv2_3_2", "conv2_3_2", 0);
  add_conv_layer(net, "conv2_3_3_conv", "conv2_3_2", "conv2_3_3", NULL, NULL, 1, 64, 1, 1, 1, 1, 0, 0, 1, 0);
  { const char* const names[] = { "conv2_3_3", "conv2_2" }; add_eltwise_layer(net, "conv2_3", names, "conv2_3", 2); }
  add_conv_layer(net, "conv3_1_1_conv", "conv2_3", "conv3_1_1", NULL, NULL, 1, 48, 1, 1, 2, 2, 0, 0, 1, 0);
  add_relu_layer(net, "conv3_1_1_relu", "conv3_1_1", "conv3_1_1", 0);
  add_conv_layer(net, "conv3_1_2_conv", "conv3_1_1", "conv3_1_2_conv", NULL, NULL, 1, 48, 3, 3, 1, 1, 1, 1, 1, 0);
  add_scale_const_layer(net, "conv3_1_2_neg", "conv3_1_2_conv", "conv3_1_2_neg", -1.000000, 0.000000, 1);
  { const char* const names[] = { "conv3_1_2_conv", "conv3_1_2_neg" }; add_concat_layer(net, "conv3_1_2_concat", names, "conv3_1_2", 2); }
  add_scale_channel_layer(net, "conv3_1_2_scale", "conv3_1_2", "conv3_1_2", NULL, NULL, 1);
  add_relu_layer(net, "conv3_1_2_relu", "conv3_1_2", "conv3_1_2", 0);
  add_conv_layer(net, "conv3_1_3_conv", "conv3_1_2", "conv3_1_3", NULL, NULL, 1, 128, 1, 1, 1, 1, 0, 0, 1, 0);
  add_conv_layer(net, "conv3_1_proj", "conv2_3", "conv3_1_proj", NULL, NULL, 1, 128, 1, 1, 2, 2, 0, 0, 1, 0);
  { const char* const names[] = { "conv3_1_3", "conv3_1_proj" }; add_eltwise_layer(net, "conv3_1", names, "conv3_1", 2); }
  add_conv_layer(net, "conv3_2_1_conv", "conv3_1", "conv3_2_1", NULL, NULL, 1, 48, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv3_2_1_relu", "conv3_2_1", "conv3_2_1", 0);
  add_conv_layer(net, "conv3_2_2_conv", "conv3_2_1", "conv3_2_2_conv", NULL, NULL, 1, 48, 3, 3, 1, 1, 1, 1, 1, 0);
  add_scale_const_layer(net, "conv3_2_2_neg", "conv3_2_2_conv", "conv3_2_2_neg", -1.000000, 0.000000, 1);
  { const char* const names[] = { "conv3_2_2_conv", "conv3_2_2_neg" }; add_concat_layer(net, "conv3_2_2_concat", names, "conv3_2_2", 2); }
  add_scale_channel_layer(net, "conv3_2_2_scale", "conv3_2_2", "conv3_2_2", NULL, NULL, 1);
  add_relu_layer(net, "conv3_2_2_relu", "conv3_2_2", "conv3_2_2", 0);
  add_conv_layer(net, "conv3_2_3_conv", "conv3_2_2", "conv3_2_3", NULL, NULL, 1, 128, 1, 1, 1, 1, 0, 0, 1, 0);
  { const char* const names[] = { "conv3_2_3", "conv3_1" }; add_eltwise_layer(net, "conv3_2", names, "conv3_2", 2); }
  add_conv_layer(net, "conv3_3_1_conv", "conv3_2", "conv3_3_1", NULL, NULL, 1, 48, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv3_3_1_relu", "conv3_3_1", "conv3_3_1", 0);
  add_conv_layer(net, "conv3_3_2_conv", "conv3_3_1", "conv3_3_2_conv", NULL, NULL, 1, 48, 3, 3, 1, 1, 1, 1, 1, 0);
  add_scale_const_layer(net, "conv3_3_2_neg", "conv3_3_2_conv", "conv3_3_2_neg", -1.000000, 0.000000, 1);
  { const char* const names[] = { "conv3_3_2_conv", "conv3_3_2_neg" }; add_concat_layer(net, "conv3_3_2_concat", names, "conv3_3_2", 2); }
  add_scale_channel_layer(net, "conv3_3_2_scale", "conv3_3_2", "conv3_3_2", NULL, NULL, 1);
  add_relu_layer(net, "conv3_3_2_relu", "conv3_3_2", "conv3_3_2", 0);
  add_conv_layer(net, "conv3_3_3_conv", "conv3_3_2", "conv3_3_3", NULL, NULL, 1, 128, 1, 1, 1, 1, 0, 0, 1, 0);
  { const char* const names[] = { "conv3_3_3", "conv3_2" }; add_eltwise_layer(net, "conv3_3", names, "conv3_3", 2); }
  add_conv_layer(net, "conv3_4_1_conv", "conv3_3", "conv3_4_1", NULL, NULL, 1, 48, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv3_4_1_relu", "conv3_4_1", "conv3_4_1", 0);
  add_conv_layer(net, "conv3_4_2_conv", "conv3_4_1", "conv3_4_2_conv", NULL, NULL, 1, 48, 3, 3, 1, 1, 1, 1, 1, 0);
  add_scale_const_layer(net, "conv3_4_2_neg", "conv3_4_2_conv", "conv3_4_2_neg", -1.000000, 0.000000, 1);
  { const char* const names[] = { "conv3_4_2_conv", "conv3_4_2_neg" }; add_concat_layer(net, "conv3_4_2_concat", names, "conv3_4_2", 2); }
  add_scale_channel_layer(net, "conv3_4_2_scale", "conv3_4_2", "conv3_4_2", NULL, NULL, 1);
  add_relu_layer(net, "conv3_4_2_relu", "conv3_4_2", "conv3_4_2", 0);
  add_conv_layer(net, "conv3_4_3_conv", "conv3_4_2", "conv3_4_3", NULL, NULL, 1, 128, 1, 1, 1, 1, 0, 0, 1, 0);
  { const char* const names[] = { "conv3_4_3", "conv3_3" }; add_eltwise_layer(net, "conv3_4", names, "conv3_4", 2); }
  add_conv_layer(net, "conv4_1_incep_0_conv", "conv3_4", "conv4_1_incep_0", NULL, NULL, 1, 64, 1, 1, 2, 2, 0, 0, 1, 0);
  add_relu_layer(net, "conv4_1_incep_0_relu", "conv4_1_incep_0", "conv4_1_incep_0", 0);
  add_conv_layer(net, "conv4_1_incep_1_reduce_conv", "conv3_4", "conv4_1_incep_1_reduce", NULL, NULL, 1, 48, 1, 1, 2, 2, 0, 0, 1, 0);
  add_relu_layer(net, "conv4_1_incep_1_reduce_relu", "conv4_1_incep_1_reduce", "conv4_1_incep_1_reduce", 0);
  add_conv_layer(net, "conv4_1_incep_1_0_conv", "conv4_1_incep_1_reduce", "conv4_1_incep_1_0", NULL, NULL, 1, 128, 3, 3, 1, 1, 1, 1, 1, 0);
  add_relu_layer(net, "conv4_1_incep_1_0_relu", "conv4_1_incep_1_0", "conv4_1_incep_1_0", 0);
  add_conv_layer(net, "conv4_1_incep_2_reduce_conv", "conv3_4", "conv4_1_incep_2_reduce", NULL, NULL, 1, 24, 1, 1, 2, 2, 0, 0, 1, 0);
  add_relu_layer(net, "conv4_1_incep_2_reduce_relu", "conv4_1_incep_2_reduce", "conv4_1_incep_2_reduce", 0);
  add_conv_layer(net, "conv4_1_incep_2_0_conv", "conv4_1_incep_2_reduce", "conv4_1_incep_2_0", NULL, NULL, 1, 48, 3, 3, 1, 1, 1, 1, 1, 0);
  add_relu_layer(net, "conv4_1_incep_2_0_relu", "conv4_1_incep_2_0", "conv4_1_incep_2_0", 0);
  add_conv_layer(net, "conv4_1_incep_2_1_conv", "conv4_1_incep_2_0", "conv4_1_incep_2_1", NULL, NULL, 1, 48, 3, 3, 1, 1, 1, 1, 1, 0);
  add_relu_layer(net, "conv4_1_incep_2_1_relu", "conv4_1_incep_2_1", "conv4_1_incep_2_1", 0);
  add_pool_layer(net, "conv4_1_incep_pool", "conv3_4", "conv4_1_incep_pool", 3, 3, 2, 2, 0, 0);
  add_conv_layer(net, "conv4_1_incep_poolproj_conv", "conv4_1_incep_pool", "conv4_1_incep_poolproj", NULL, NULL, 1, 128, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv4_1_incep_poolproj_relu", "conv4_1_incep_poolproj", "conv4_1_incep_poolproj", 0);
  { const char* const names[] = { "conv4_1_incep_0", "conv4_1_incep_1_0", "conv4_1_incep_2_1", "conv4_1_incep_poolproj" }; add_concat_layer(net, "conv4_1_incep", names, "conv4_1_incep", 4); }
  add_conv_layer(net, "conv4_1_out_conv", "conv4_1_incep", "conv4_1_out", NULL, NULL, 1, 256, 1, 1, 1, 1, 0, 0, 1, 0);
  add_conv_layer(net, "conv4_1_proj", "conv3_4", "conv4_1_proj", NULL, NULL, 1, 256, 1, 1, 2, 2, 0, 0, 1, 0);
  { const char* const names[] = { "conv4_1_out", "conv4_1_proj" }; add_eltwise_layer(net, "conv4_1", names, "conv4_1", 2); }
  add_conv_layer(net, "conv4_2_incep_0_conv", "conv4_1", "conv4_2_incep_0", NULL, NULL, 1, 64, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv4_2_incep_0_relu", "conv4_2_incep_0", "conv4_2_incep_0", 0);
  add_conv_layer(net, "conv4_2_incep_1_reduce_conv", "conv4_1", "conv4_2_incep_1_reduce", NULL, NULL, 1, 64, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv4_2_incep_1_reduce_relu", "conv4_2_incep_1_reduce", "conv4_2_incep_1_reduce", 0);
  add_conv_layer(net, "conv4_2_incep_1_0_conv", "conv4_2_incep_1_reduce", "conv4_2_incep_1_0", NULL, NULL, 1, 128, 3, 3, 1, 1, 1, 1, 1, 0);
  add_relu_layer(net, "conv4_2_incep_1_0_relu", "conv4_2_incep_1_0", "conv4_2_incep_1_0", 0);
  add_conv_layer(net, "conv4_2_incep_2_reduce_conv", "conv4_1", "conv4_2_incep_2_reduce", NULL, NULL, 1, 24, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv4_2_incep_2_reduce_relu", "conv4_2_incep_2_reduce", "conv4_2_incep_2_reduce", 0);
  add_conv_layer(net, "conv4_2_incep_2_0_conv", "conv4_2_incep_2_reduce", "conv4_2_incep_2_0", NULL, NULL, 1, 48, 3, 3, 1, 1, 1, 1, 1, 0);
  add_relu_layer(net, "conv4_2_incep_2_0_relu", "conv4_2_incep_2_0", "conv4_2_incep_2_0", 0);
  add_conv_layer(net, "conv4_2_incep_2_1_conv", "conv4_2_incep_2_0", "conv4_2_incep_2_1", NULL, NULL, 1, 48, 3, 3, 1, 1, 1, 1, 1, 0);
  add_relu_layer(net, "conv4_2_incep_2_1_relu", "conv4_2_incep_2_1", "conv4_2_incep_2_1", 0);
  { const char* const names[] = { "conv4_2_incep_0", "conv4_2_incep_1_0", "conv4_2_incep_2_1" }; add_concat_layer(net, "conv4_2_incep", names, "conv4_2_incep", 3); }
  add_conv_layer(net, "conv4_2_out_conv", "conv4_2_incep", "conv4_2_out", NULL, NULL, 1, 256, 1, 1, 1, 1, 0, 0, 1, 0);
  { const char* const names[] = { "conv4_2_out", "conv4_1" }; add_eltwise_layer(net, "conv4_2", names, "conv4_2", 2); }
  add_conv_layer(net, "conv4_3_incep_0_conv", "conv4_2", "conv4_3_incep_0", NULL, NULL, 1, 64, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv4_3_incep_0_relu", "conv4_3_incep_0", "conv4_3_incep_0", 0);
  add_conv_layer(net, "conv4_3_incep_1_reduce_conv", "conv4_2", "conv4_3_incep_1_reduce", NULL, NULL, 1, 64, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv4_3_incep_1_reduce_relu", "conv4_3_incep_1_reduce", "conv4_3_incep_1_reduce", 0);
  add_conv_layer(net, "conv4_3_incep_1_0_conv", "conv4_3_incep_1_reduce", "conv4_3_incep_1_0", NULL, NULL, 1, 128, 3, 3, 1, 1, 1, 1, 1, 0);
  add_relu_layer(net, "conv4_3_incep_1_0_relu", "conv4_3_incep_1_0", "conv4_3_incep_1_0", 0);
  add_conv_layer(net, "conv4_3_incep_2_reduce_conv", "conv4_2", "conv4_3_incep_2_reduce", NULL, NULL, 1, 24, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv4_3_incep_2_reduce_relu", "conv4_3_incep_2_reduce", "conv4_3_incep_2_reduce", 0);
  add_conv_layer(net, "conv4_3_incep_2_0_conv", "conv4_3_incep_2_reduce", "conv4_3_incep_2_0", NULL, NULL, 1, 48, 3, 3, 1, 1, 1, 1, 1, 0);
  add_relu_layer(net, "conv4_3_incep_2_0_relu", "conv4_3_incep_2_0", "conv4_3_incep_2_0", 0);
  add_conv_layer(net, "conv4_3_incep_2_1_conv", "conv4_3_incep_2_0", "conv4_3_incep_2_1", NULL, NULL, 1, 48, 3, 3, 1, 1, 1, 1, 1, 0);
  add_relu_layer(net, "conv4_3_incep_2_1_relu", "conv4_3_incep_2_1", "conv4_3_incep_2_1", 0);
  { const char* const names[] = { "conv4_3_incep_0", "conv4_3_incep_1_0", "conv4_3_incep_2_1" }; add_concat_layer(net, "conv4_3_incep", names, "conv4_3_incep", 3); }
  add_conv_layer(net, "conv4_3_out_conv", "conv4_3_incep", "conv4_3_out", NULL, NULL, 1, 256, 1, 1, 1, 1, 0, 0, 1, 0);
  { const char* const names[] = { "conv4_3_out", "conv4_2" }; add_eltwise_layer(net, "conv4_3", names, "conv4_3", 2); }
  add_conv_layer(net, "conv4_4_incep_0_conv", "conv4_3", "conv4_4_incep_0", NULL, NULL, 1, 64, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv4_4_incep_0_relu", "conv4_4_incep_0", "conv4_4_incep_0", 0);
  add_conv_layer(net, "conv4_4_incep_1_reduce_conv", "conv4_3", "conv4_4_incep_1_reduce", NULL, NULL, 1, 64, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv4_4_incep_1_reduce_relu", "conv4_4_incep_1_reduce", "conv4_4_incep_1_reduce", 0);
  add_conv_layer(net, "conv4_4_incep_1_0_conv", "conv4_4_incep_1_reduce", "conv4_4_incep_1_0", NULL, NULL, 1, 128, 3, 3, 1, 1, 1, 1, 1, 0);
  add_relu_layer(net, "conv4_4_incep_1_0_relu", "conv4_4_incep_1_0", "conv4_4_incep_1_0", 0);
  add_conv_layer(net, "conv4_4_incep_2_reduce_conv", "conv4_3", "conv4_4_incep_2_reduce", NULL, NULL, 1, 24, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv4_4_incep_2_reduce_relu", "conv4_4_incep_2_reduce", "conv4_4_incep_2_reduce", 0);
  add_conv_layer(net, "conv4_4_incep_2_0_conv", "conv4_4_incep_2_reduce", "conv4_4_incep_2_0", NULL, NULL, 1, 48, 3, 3, 1, 1, 1, 1, 1, 0);
  add_relu_layer(net, "conv4_4_incep_2_0_relu", "conv4_4_incep_2_0", "conv4_4_incep_2_0", 0);
  add_conv_layer(net, "conv4_4_incep_2_1_conv", "conv4_4_incep_2_0", "conv4_4_incep_2_1", NULL, NULL, 1, 48, 3, 3, 1, 1, 1, 1, 1, 0);
  add_relu_layer(net, "conv4_4_incep_2_1_relu", "conv4_4_incep_2_1", "conv4_4_incep_2_1", 0);
  { const char* const names[] = { "conv4_4_incep_0", "conv4_4_incep_1_0", "conv4_4_incep_2_1" }; add_concat_layer(net, "conv4_4_incep", names, "conv4_4_incep", 3); }
  add_conv_layer(net, "conv4_4_out_conv", "conv4_4_incep", "conv4_4_out", NULL, NULL, 1, 256, 1, 1, 1, 1, 0, 0, 1, 0);
  { const char* const names[] = { "conv4_4_out", "conv4_3" }; add_eltwise_layer(net, "conv4_4", names, "conv4_4", 2); }
  add_conv_layer(net, "conv5_1_incep_0_conv", "conv4_4", "conv5_1_incep_0", NULL, NULL, 1, 64, 1, 1, 2, 2, 0, 0, 1, 0);
  add_relu_layer(net, "conv5_1_incep_0_relu", "conv5_1_incep_0", "conv5_1_incep_0", 0);
  add_conv_layer(net, "conv5_1_incep_1_reduce_conv", "conv4_4", "conv5_1_incep_1_reduce", NULL, NULL, 1, 96, 1, 1, 2, 2, 0, 0, 1, 0);
  add_relu_layer(net, "conv5_1_incep_1_reduce_relu", "conv5_1_incep_1_reduce", "conv5_1_incep_1_reduce", 0);
  add_conv_layer(net, "conv5_1_incep_1_0_conv", "conv5_1_incep_1_reduce", "conv5_1_incep_1_0", NULL, NULL, 1, 192, 3, 3, 1, 1, 1, 1, 1, 0);
  add_relu_layer(net, "conv5_1_incep_1_0_relu", "conv5_1_incep_1_0", "conv5_1_incep_1_0", 0);
  add_conv_layer(net, "conv5_1_incep_2_reduce_conv", "conv4_4", "conv5_1_incep_2_reduce", NULL, NULL, 1, 32, 1, 1, 2, 2, 0, 0, 1, 0);
  add_relu_layer(net, "conv5_1_incep_2_reduce_relu", "conv5_1_incep_2_reduce", "conv5_1_incep_2_reduce", 0);
  add_conv_layer(net, "conv5_1_incep_2_0_conv", "conv5_1_incep_2_reduce", "conv5_1_incep_2_0", NULL, NULL, 1, 64, 3, 3, 1, 1, 1, 1, 1, 0);
  add_relu_layer(net, "conv5_1_incep_2_0_relu", "conv5_1_incep_2_0", "conv5_1_incep_2_0", 0);
  add_conv_layer(net, "conv5_1_incep_2_1_conv", "conv5_1_incep_2_0", "conv5_1_incep_2_1", NULL, NULL, 1, 64, 3, 3, 1, 1, 1, 1, 1, 0);
  add_relu_layer(net, "conv5_1_incep_2_1_relu", "conv5_1_incep_2_1", "conv5_1_incep_2_1", 0);
  add_pool_layer(net, "conv5_1_incep_pool", "conv4_4", "conv5_1_incep_pool", 3, 3, 2, 2, 0, 0);
  add_conv_layer(net, "conv5_1_incep_poolproj_conv", "conv5_1_incep_pool", "conv5_1_incep_poolproj", NULL, NULL, 1, 128, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv5_1_incep_poolproj_relu", "conv5_1_incep_poolproj", "conv5_1_incep_poolproj", 0);
  { const char* const names[] = { "conv5_1_incep_0", "conv5_1_incep_1_0", "conv5_1_incep_2_1", "conv5_1_incep_poolproj" }; add_concat_layer(net, "conv5_1_incep", names, "conv5_1_incep", 4); }
  add_conv_layer(net, "conv5_1_out_conv", "conv5_1_incep", "conv5_1_out", NULL, NULL, 1, 384, 1, 1, 1, 1, 0, 0, 1, 0);
  add_conv_layer(net, "conv5_1_proj", "conv4_4", "conv5_1_proj", NULL, NULL, 1, 384, 1, 1, 2, 2, 0, 0, 1, 0);
  { const char* const names[] = { "conv5_1_out", "conv5_1_proj" }; add_eltwise_layer(net, "conv5_1", names, "conv5_1", 2); }
  add_conv_layer(net, "conv5_2_incep_0_conv", "conv5_1", "conv5_2_incep_0", NULL, NULL, 1, 64, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv5_2_incep_0_relu", "conv5_2_incep_0", "conv5_2_incep_0", 0);
  add_conv_layer(net, "conv5_2_incep_1_reduce_conv", "conv5_1", "conv5_2_incep_1_reduce", NULL, NULL, 1, 96, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv5_2_incep_1_reduce_relu", "conv5_2_incep_1_reduce", "conv5_2_incep_1_reduce", 0);
  add_conv_layer(net, "conv5_2_incep_1_0_conv", "conv5_2_incep_1_reduce", "conv5_2_incep_1_0", NULL, NULL, 1, 192, 3, 3, 1, 1, 1, 1, 1, 0);
  add_relu_layer(net, "conv5_2_incep_1_0_relu", "conv5_2_incep_1_0", "conv5_2_incep_1_0", 0);
  add_conv_layer(net, "conv5_2_incep_2_reduce_conv", "conv5_1", "conv5_2_incep_2_reduce", NULL, NULL, 1, 32, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv5_2_incep_2_reduce_relu", "conv5_2_incep_2_reduce", "conv5_2_incep_2_reduce", 0);
  add_conv_layer(net, "conv5_2_incep_2_0_conv", "conv5_2_incep_2_reduce", "conv5_2_incep_2_0", NULL, NULL, 1, 64, 3, 3, 1, 1, 1, 1, 1, 0);
  add_relu_layer(net, "conv5_2_incep_2_0_relu", "conv5_2_incep_2_0", "conv5_2_incep_2_0", 0);
  add_conv_layer(net, "conv5_2_incep_2_1_conv", "conv5_2_incep_2_0", "conv5_2_incep_2_1", NULL, NULL, 1, 64, 3, 3, 1, 1, 1, 1, 1, 0);
  add_relu_layer(net, "conv5_2_incep_2_1_relu", "conv5_2_incep_2_1", "conv5_2_incep_2_1", 0);
  { const char* const names[] = { "conv5_2_incep_0", "conv5_2_incep_1_0", "conv5_2_incep_2_1" }; add_concat_layer(net, "conv5_2_incep", names, "conv5_2_incep", 3); }
  add_conv_layer(net, "conv5_2_out_conv", "conv5_2_incep", "conv5_2_out", NULL, NULL, 1, 384, 1, 1, 1, 1, 0, 0, 1, 0);
  { const char* const names[] = { "conv5_2_out", "conv5_1" }; add_eltwise_layer(net, "conv5_2", names, "conv5_2", 2); }
  add_conv_layer(net, "conv5_3_incep_0_conv", "conv5_2", "conv5_3_incep_0", NULL, NULL, 1, 64, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv5_3_incep_0_relu", "conv5_3_incep_0", "conv5_3_incep_0", 0);
  add_conv_layer(net, "conv5_3_incep_1_reduce_conv", "conv5_2", "conv5_3_incep_1_reduce", NULL, NULL, 1, 96, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv5_3_incep_1_reduce_relu", "conv5_3_incep_1_reduce", "conv5_3_incep_1_reduce", 0);
  add_conv_layer(net, "conv5_3_incep_1_0_conv", "conv5_3_incep_1_reduce", "conv5_3_incep_1_0", NULL, NULL, 1, 192, 3, 3, 1, 1, 1, 1, 1, 0);
  add_relu_layer(net, "conv5_3_incep_1_0_relu", "conv5_3_incep_1_0", "conv5_3_incep_1_0", 0);
  add_conv_layer(net, "conv5_3_incep_2_reduce_conv", "conv5_2", "conv5_3_incep_2_reduce", NULL, NULL, 1, 32, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv5_3_incep_2_reduce_relu", "conv5_3_incep_2_reduce", "conv5_3_incep_2_reduce", 0);
  add_conv_layer(net, "conv5_3_incep_2_0_conv", "conv5_3_incep_2_reduce", "conv5_3_incep_2_0", NULL, NULL, 1, 64, 3, 3, 1, 1, 1, 1, 1, 0);
  add_relu_layer(net, "conv5_3_incep_2_0_relu", "conv5_3_incep_2_0", "conv5_3_incep_2_0", 0);
  add_conv_layer(net, "conv5_3_incep_2_1_conv", "conv5_3_incep_2_0", "conv5_3_incep_2_1", NULL, NULL, 1, 64, 3, 3, 1, 1, 1, 1, 1, 0);
  add_relu_layer(net, "conv5_3_incep_2_1_relu", "conv5_3_incep_2_1", "conv5_3_incep_2_1", 0);
  { const char* const names[] = { "conv5_3_incep_0", "conv5_3_incep_1_0", "conv5_3_incep_2_1" }; add_concat_layer(net, "conv5_3_incep", names, "conv5_3_incep", 3); }
  add_conv_layer(net, "conv5_3_out_conv", "conv5_3_incep", "conv5_3_out", NULL, NULL, 1, 384, 1, 1, 1, 1, 0, 0, 1, 0);
  { const char* const names[] = { "conv5_3_out", "conv5_2" }; add_eltwise_layer(net, "conv5_3", names, "conv5_3", 2); }
  add_conv_layer(net, "conv5_4_incep_0_conv", "conv5_3", "conv5_4_incep_0", NULL, NULL, 1, 64, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv5_4_incep_0_relu", "conv5_4_incep_0", "conv5_4_incep_0", 0);
  add_conv_layer(net, "conv5_4_incep_1_reduce_conv", "conv5_3", "conv5_4_incep_1_reduce", NULL, NULL, 1, 96, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv5_4_incep_1_reduce_relu", "conv5_4_incep_1_reduce", "conv5_4_incep_1_reduce", 0);
  add_conv_layer(net, "conv5_4_incep_1_0_conv", "conv5_4_incep_1_reduce", "conv5_4_incep_1_0", NULL, NULL, 1, 192, 3, 3, 1, 1, 1, 1, 1, 0);
  add_relu_layer(net, "conv5_4_incep_1_0_relu", "conv5_4_incep_1_0", "conv5_4_incep_1_0", 0);
  add_conv_layer(net, "conv5_4_incep_2_reduce_conv", "conv5_3", "conv5_4_incep_2_reduce", NULL, NULL, 1, 32, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "conv5_4_incep_2_reduce_relu", "conv5_4_incep_2_reduce", "conv5_4_incep_2_reduce", 0);
  add_conv_layer(net, "conv5_4_incep_2_0_conv", "conv5_4_incep_2_reduce", "conv5_4_incep_2_0", NULL, NULL, 1, 64, 3, 3, 1, 1, 1, 1, 1, 0);
  add_relu_layer(net, "conv5_4_incep_2_0_relu", "conv5_4_incep_2_0", "conv5_4_incep_2_0", 0);
  add_conv_layer(net, "conv5_4_incep_2_1_conv", "conv5_4_incep_2_0", "conv5_4_incep_2_1", NULL, NULL, 1, 64, 3, 3, 1, 1, 1, 1, 1, 0);
  add_relu_layer(net, "conv5_4_incep_2_1_relu", "conv5_4_incep_2_1", "conv5_4_incep_2_1", 0);
  { const char* const names[] = { "conv5_4_incep_0", "conv5_4_incep_1_0", "conv5_4_incep_2_1" }; add_concat_layer(net, "conv5_4_incep", names, "conv5_4_incep", 3); }
  add_conv_layer(net, "conv5_4_out_conv", "conv5_4_incep", "conv5_4_out", NULL, NULL, 1, 384, 1, 1, 1, 1, 0, 0, 1, 0);
  { const char* const names[] = { "conv5_4_out", "conv5_3" }; add_eltwise_layer(net, "conv5_4", names, "conv5_4", 2); }
  add_pool_layer(net, "downsample", "conv3_4", "downsample", 3, 3, 2, 2, 0, 0);
  add_deconv_layer(net, "upsample", "conv5_4", "upsample", NULL, NULL, 384, 384, 4, 4, 2, 2, 1, 1, 1, 0);
  { const char* const names[] = { "downsample", "conv4_4", "upsample" }; add_concat_layer(net, "concat", names, "concat", 3); }
  add_conv_layer(net, "convf_rpn", "concat", "convf_rpn", NULL, NULL, 1, 128, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "reluf_rpn", "convf_rpn", "convf_rpn", 0);
  add_conv_layer(net, "convf_2", "concat", "convf_2", NULL, NULL, 1, 384, 1, 1, 1, 1, 0, 0, 1, 0);
  add_relu_layer(net, "reluf_2", "convf_2", "convf_2", 0);
  { const char* const names[] = { "convf_rpn", "convf_2" }; add_concat_layer(net, "concat_convf", names, "convf", 2); }
}
