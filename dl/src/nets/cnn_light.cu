#include "nets/net_factory.h"

void setup_shared_cnn_light(Net* const net, const int input_size)
{
  add_image_layer(net, "input-data", "data", "im_info", input_size, 32, 2048);
  add_conv_layer(net, "conv1", "data", "conv1", NULL, NULL, 1, 32, 4, 4, 2, 2, 1, 1, 1);
  add_relu_layer(net, "relu1", "conv1", "conv1", 0);
  add_conv_layer(net, "conv2", "conv1", "conv2", NULL, NULL, 1, 48, 3, 3, 2, 2, 1, 1, 1);
  add_relu_layer(net, "relu2", "conv2", "conv2", 0);
  add_conv_layer(net, "conv3", "conv2", "conv3", NULL, NULL, 1, 96, 3, 3, 2, 2, 1, 1, 1);
  add_relu_layer(net, "relu3", "conv3", "conv3", 0);
  add_pool_layer(net, "inc3a_pool1", "conv3", "inc3a_pool1", 3, 3, 2, 2, 0, 0);
  add_conv_layer(net, "inc3a_conv1", "inc3a_pool1", "inc3a_conv1", NULL, NULL, 1, 96, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc3a_relu1", "inc3a_conv1", "inc3a_conv1", 0);
  add_conv_layer(net, "inc3a_conv3_1", "conv3", "inc3a_conv3_1", NULL, NULL, 1, 16, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc3a_relu3_1", "inc3a_conv3_1", "inc3a_conv3_1", 0);
  add_conv_layer(net, "inc3a_conv3_2", "inc3a_conv3_1", "inc3a_conv3_2", NULL, NULL, 1, 64, 3, 3, 2, 2, 1, 1, 1);
  add_relu_layer(net, "inc3a_relu3_2", "inc3a_conv3_2", "inc3a_conv3_2", 0);
  add_conv_layer(net, "inc3a_conv5_1", "conv3", "inc3a_conv5_1", NULL, NULL, 1, 16, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc3a_relu5_1", "inc3a_conv5_1", "inc3a_conv5_1", 0);
  add_conv_layer(net, "inc3a_conv5_2", "inc3a_conv5_1", "inc3a_conv5_2", NULL, NULL, 1, 32, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc3a_relu5_2", "inc3a_conv5_2", "inc3a_conv5_2", 0);
  add_conv_layer(net, "inc3a_conv5_3", "inc3a_conv5_2", "inc3a_conv5_3", NULL, NULL, 1, 32, 3, 3, 2, 2, 1, 1, 1);
  add_relu_layer(net, "inc3a_relu5_3", "inc3a_conv5_3", "inc3a_conv5_3", 0);
  { const char* const names[] = { "inc3a_conv1", "inc3a_conv3_2", "inc3a_conv5_3" }; add_concat_layer(net, "inc3a", names, "inc3a", 3); }
  add_conv_layer(net, "inc3b_conv1", "inc3a", "inc3b_conv1", NULL, NULL, 1, 96, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc3b_relu1", "inc3b_conv1", "inc3b_conv1", 0);
  add_conv_layer(net, "inc3b_conv3_1", "inc3a", "inc3b_conv3_1", NULL, NULL, 1, 16, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc3b_relu3_1", "inc3b_conv3_1", "inc3b_conv3_1", 0);
  add_conv_layer(net, "inc3b_conv3_2", "inc3b_conv3_1", "inc3b_conv3_2", NULL, NULL, 1, 64, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc3b_relu3_2", "inc3b_conv3_2", "inc3b_conv3_2", 0);
  add_conv_layer(net, "inc3b_conv5_1", "inc3a", "inc3b_conv5_1", NULL, NULL, 1, 16, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc3b_relu5_1", "inc3b_conv5_1", "inc3b_conv5_1", 0);
  add_conv_layer(net, "inc3b_conv5_2", "inc3b_conv5_1", "inc3b_conv5_2", NULL, NULL, 1, 32, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc3b_relu5_2", "inc3b_conv5_2", "inc3b_conv5_2", 0);
  add_conv_layer(net, "inc3b_conv5_3", "inc3b_conv5_2", "inc3b_conv5_3", NULL, NULL, 1, 32, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc3b_relu5_3", "inc3b_conv5_3", "inc3b_conv5_3", 0);
  { const char* const names[] = { "inc3b_conv1", "inc3b_conv3_2", "inc3b_conv5_3" }; add_concat_layer(net, "inc3b", names, "inc3b", 3); }
  add_conv_layer(net, "inc3c_conv1", "inc3b", "inc3c_conv1", NULL, NULL, 1, 96, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc3c_relu1", "inc3c_conv1", "inc3c_conv1", 0);
  add_conv_layer(net, "inc3c_conv3_1", "inc3b", "inc3c_conv3_1", NULL, NULL, 1, 16, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc3c_relu3_1", "inc3c_conv3_1", "inc3c_conv3_1", 0);
  add_conv_layer(net, "inc3c_conv3_2", "inc3c_conv3_1", "inc3c_conv3_2", NULL, NULL, 1, 64, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc3c_relu3_2", "inc3c_conv3_2", "inc3c_conv3_2", 0);
  add_conv_layer(net, "inc3c_conv5_1", "inc3b", "inc3c_conv5_1", NULL, NULL, 1, 16, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc3c_relu5_1", "inc3c_conv5_1", "inc3c_conv5_1", 0);
  add_conv_layer(net, "inc3c_conv5_2", "inc3c_conv5_1", "inc3c_conv5_2", NULL, NULL, 1, 32, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc3c_relu5_2", "inc3c_conv5_2", "inc3c_conv5_2", 0);
  add_conv_layer(net, "inc3c_conv5_3", "inc3c_conv5_2", "inc3c_conv5_3", NULL, NULL, 1, 32, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc3c_relu5_3", "inc3c_conv5_3", "inc3c_conv5_3", 0);
  { const char* const names[] = { "inc3c_conv1", "inc3c_conv3_2", "inc3c_conv5_3" }; add_concat_layer(net, "inc3c", names, "inc3c", 3); }
  add_conv_layer(net, "inc3d_conv1", "inc3c", "inc3d_conv1", NULL, NULL, 1, 96, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc3d_relu1", "inc3d_conv1", "inc3d_conv1", 0);
  add_conv_layer(net, "inc3d_conv3_1", "inc3c", "inc3d_conv3_1", NULL, NULL, 1, 16, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc3d_relu3_1", "inc3d_conv3_1", "inc3d_conv3_1", 0);
  add_conv_layer(net, "inc3d_conv3_2", "inc3d_conv3_1", "inc3d_conv3_2", NULL, NULL, 1, 64, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc3d_relu3_2", "inc3d_conv3_2", "inc3d_conv3_2", 0);
  add_conv_layer(net, "inc3d_conv5_1", "inc3c", "inc3d_conv5_1", NULL, NULL, 1, 16, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc3d_relu5_1", "inc3d_conv5_1", "inc3d_conv5_1", 0);
  add_conv_layer(net, "inc3d_conv5_2", "inc3d_conv5_1", "inc3d_conv5_2", NULL, NULL, 1, 32, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc3d_relu5_2", "inc3d_conv5_2", "inc3d_conv5_2", 0);
  add_conv_layer(net, "inc3d_conv5_3", "inc3d_conv5_2", "inc3d_conv5_3", NULL, NULL, 1, 32, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc3d_relu5_3", "inc3d_conv5_3", "inc3d_conv5_3", 0);
  { const char* const names[] = { "inc3d_conv1", "inc3d_conv3_2", "inc3d_conv5_3" }; add_concat_layer(net, "inc3d", names, "inc3d", 3); }
  add_conv_layer(net, "inc3e_conv1", "inc3d", "inc3e_conv1", NULL, NULL, 1, 96, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc3e_relu1", "inc3e_conv1", "inc3e_conv1", 0);
  add_conv_layer(net, "inc3e_conv3_1", "inc3d", "inc3e_conv3_1", NULL, NULL, 1, 16, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc3e_relu3_1", "inc3e_conv3_1", "inc3e_conv3_1", 0);
  add_conv_layer(net, "inc3e_conv3_2", "inc3e_conv3_1", "inc3e_conv3_2", NULL, NULL, 1, 64, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc3e_relu3_2", "inc3e_conv3_2", "inc3e_conv3_2", 0);
  add_conv_layer(net, "inc3e_conv5_1", "inc3d", "inc3e_conv5_1", NULL, NULL, 1, 16, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc3e_relu5_1", "inc3e_conv5_1", "inc3e_conv5_1", 0);
  add_conv_layer(net, "inc3e_conv5_2", "inc3e_conv5_1", "inc3e_conv5_2", NULL, NULL, 1, 32, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc3e_relu5_2", "inc3e_conv5_2", "inc3e_conv5_2", 0);
  add_conv_layer(net, "inc3e_conv5_3", "inc3e_conv5_2", "inc3e_conv5_3", NULL, NULL, 1, 32, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc3e_relu5_3", "inc3e_conv5_3", "inc3e_conv5_3", 0);
  { const char* const names[] = { "inc3e_conv1", "inc3e_conv3_2", "inc3e_conv5_3" }; add_concat_layer(net, "inc3e", names, "inc3e", 3); }
  add_pool_layer(net, "inc4a_pool1", "inc3e", "inc4a_pool1", 3, 3, 2, 2, 0, 0);
  add_conv_layer(net, "inc4a_conv1", "inc4a_pool1", "inc4a_conv1", NULL, NULL, 1, 128, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc4a_relu1", "inc4a_conv1", "inc4a_conv1", 0);
  add_conv_layer(net, "inc4a_conv3_1", "inc3e", "inc4a_conv3_1", NULL, NULL, 1, 32, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc4a_relu3_1", "inc4a_conv3_1", "inc4a_conv3_1", 0);
  add_conv_layer(net, "inc4a_conv3_2", "inc4a_conv3_1", "inc4a_conv3_2", NULL, NULL, 1, 96, 3, 3, 2, 2, 1, 1, 1);
  add_relu_layer(net, "inc4a_relu3_2", "inc4a_conv3_2", "inc4a_conv3_2", 0);
  add_conv_layer(net, "inc4a_conv5_1", "inc3e", "inc4a_conv5_1", NULL, NULL, 1, 16, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc4a_relu5_1", "inc4a_conv5_1", "inc4a_conv5_1", 0);
  add_conv_layer(net, "inc4a_conv5_2", "inc4a_conv5_1", "inc4a_conv5_2", NULL, NULL, 1, 32, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc4a_relu5_2", "inc4a_conv5_2", "inc4a_conv5_2", 0);
  add_conv_layer(net, "inc4a_conv5_3", "inc4a_conv5_2", "inc4a_conv5_3", NULL, NULL, 1, 32, 3, 3, 2, 2, 1, 1, 1);
  add_relu_layer(net, "inc4a_relu5_3", "inc4a_conv5_3", "inc4a_conv5_3", 0);
  { const char* const names[] = { "inc4a_conv1", "inc4a_conv3_2", "inc4a_conv5_3" }; add_concat_layer(net, "inc4a", names, "inc4a", 3); }
  add_conv_layer(net, "inc4b_conv1", "inc4a", "inc4b_conv1", NULL, NULL, 1, 128, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc4b_relu1", "inc4b_conv1", "inc4b_conv1", 0);
  add_conv_layer(net, "inc4b_conv3_1", "inc4a", "inc4b_conv3_1", NULL, NULL, 1, 32, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc4b_relu3_1", "inc4b_conv3_1", "inc4b_conv3_1", 0);
  add_conv_layer(net, "inc4b_conv3_2", "inc4b_conv3_1", "inc4b_conv3_2", NULL, NULL, 1, 96, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc4b_relu3_2", "inc4b_conv3_2", "inc4b_conv3_2", 0);
  add_conv_layer(net, "inc4b_conv5_1", "inc4a", "inc4b_conv5_1", NULL, NULL, 1, 16, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc4b_relu5_1", "inc4b_conv5_1", "inc4b_conv5_1", 0);
  add_conv_layer(net, "inc4b_conv5_2", "inc4b_conv5_1", "inc4b_conv5_2", NULL, NULL, 1, 32, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc4b_relu5_2", "inc4b_conv5_2", "inc4b_conv5_2", 0);
  add_conv_layer(net, "inc4b_conv5_3", "inc4b_conv5_2", "inc4b_conv5_3", NULL, NULL, 1, 32, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc4b_relu5_3", "inc4b_conv5_3", "inc4b_conv5_3", 0);
  { const char* const names[] = { "inc4b_conv1", "inc4b_conv3_2", "inc4b_conv5_3" }; add_concat_layer(net, "inc4b", names, "inc4b", 3); }
  add_conv_layer(net, "inc4c_conv1", "inc4b", "inc4c_conv1", NULL, NULL, 1, 128, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc4c_relu1", "inc4c_conv1", "inc4c_conv1", 0);
  add_conv_layer(net, "inc4c_conv3_1", "inc4b", "inc4c_conv3_1", NULL, NULL, 1, 32, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc4c_relu3_1", "inc4c_conv3_1", "inc4c_conv3_1", 0);
  add_conv_layer(net, "inc4c_conv3_2", "inc4c_conv3_1", "inc4c_conv3_2", NULL, NULL, 1, 96, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc4c_relu3_2", "inc4c_conv3_2", "inc4c_conv3_2", 0);
  add_conv_layer(net, "inc4c_conv5_1", "inc4b", "inc4c_conv5_1", NULL, NULL, 1, 16, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc4c_relu5_1", "inc4c_conv5_1", "inc4c_conv5_1", 0);
  add_conv_layer(net, "inc4c_conv5_2", "inc4c_conv5_1", "inc4c_conv5_2", NULL, NULL, 1, 32, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc4c_relu5_2", "inc4c_conv5_2", "inc4c_conv5_2", 0);
  add_conv_layer(net, "inc4c_conv5_3", "inc4c_conv5_2", "inc4c_conv5_3", NULL, NULL, 1, 32, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc4c_relu5_3", "inc4c_conv5_3", "inc4c_conv5_3", 0);
  { const char* const names[] = { "inc4c_conv1", "inc4c_conv3_2", "inc4c_conv5_3" }; add_concat_layer(net, "inc4c", names, "inc4c", 3); }
  add_conv_layer(net, "inc4d_conv1", "inc4c", "inc4d_conv1", NULL, NULL, 1, 128, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc4d_relu1", "inc4d_conv1", "inc4d_conv1", 0);
  add_conv_layer(net, "inc4d_conv3_1", "inc4c", "inc4d_conv3_1", NULL, NULL, 1, 32, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc4d_relu3_1", "inc4d_conv3_1", "inc4d_conv3_1", 0);
  add_conv_layer(net, "inc4d_conv3_2", "inc4d_conv3_1", "inc4d_conv3_2", NULL, NULL, 1, 96, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc4d_relu3_2", "inc4d_conv3_2", "inc4d_conv3_2", 0);
  add_conv_layer(net, "inc4d_conv5_1", "inc4c", "inc4d_conv5_1", NULL, NULL, 1, 16, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc4d_relu5_1", "inc4d_conv5_1", "inc4d_conv5_1", 0);
  add_conv_layer(net, "inc4d_conv5_2", "inc4d_conv5_1", "inc4d_conv5_2", NULL, NULL, 1, 32, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc4d_relu5_2", "inc4d_conv5_2", "inc4d_conv5_2", 0);
  add_conv_layer(net, "inc4d_conv5_3", "inc4d_conv5_2", "inc4d_conv5_3", NULL, NULL, 1, 32, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc4d_relu5_3", "inc4d_conv5_3", "inc4d_conv5_3", 0);
  { const char* const names[] = { "inc4d_conv1", "inc4d_conv3_2", "inc4d_conv5_3" }; add_concat_layer(net, "inc4d", names, "inc4d", 3); }
  add_conv_layer(net, "inc4e_conv1", "inc4d", "inc4e_conv1", NULL, NULL, 1, 128, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc4e_relu1", "inc4e_conv1", "inc4e_conv1", 0);
  add_conv_layer(net, "inc4e_conv3_1", "inc4d", "inc4e_conv3_1", NULL, NULL, 1, 32, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc4e_relu3_1", "inc4e_conv3_1", "inc4e_conv3_1", 0);
  add_conv_layer(net, "inc4e_conv3_2", "inc4e_conv3_1", "inc4e_conv3_2", NULL, NULL, 1, 96, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc4e_relu3_2", "inc4e_conv3_2", "inc4e_conv3_2", 0);
  add_conv_layer(net, "inc4e_conv5_1", "inc4d", "inc4e_conv5_1", NULL, NULL, 1, 16, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "inc4e_relu5_1", "inc4e_conv5_1", "inc4e_conv5_1", 0);
  add_conv_layer(net, "inc4e_conv5_2", "inc4e_conv5_1", "inc4e_conv5_2", NULL, NULL, 1, 32, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc4e_relu5_2", "inc4e_conv5_2", "inc4e_conv5_2", 0);
  add_conv_layer(net, "inc4e_conv5_3", "inc4e_conv5_2", "inc4e_conv5_3", NULL, NULL, 1, 32, 3, 3, 1, 1, 1, 1, 1);
  add_relu_layer(net, "inc4e_relu5_3", "inc4e_conv5_3", "inc4e_conv5_3", 0);
  { const char* const names[] = { "inc4e_conv1", "inc4e_conv3_2", "inc4e_conv5_3" }; add_concat_layer(net, "inc4e", names, "inc4e", 3); }
  add_pool_layer(net, "downsample", "conv3", "downsample", 3, 3, 2, 2, 0, 0);
  add_deconv_layer(net, "upsample", "inc4e", "upsample", NULL, NULL, 256, 256, 4, 4, 2, 2, 1, 1, 0);
  { const char* const names[] = { "downsample", "inc3e", "upsample" }; add_concat_layer(net, "concat", names, "concat", 3); }
  add_conv_layer(net, "convf", "concat", "convf", NULL, NULL, 1, 256, 1, 1, 1, 1, 0, 0, 1);
  add_relu_layer(net, "reluf", "convf", "convf", 0);

  get_tensor_by_name(net, "conv1")->data_type = PRIVATE_DATA;
  get_tensor_by_name(net, "conv3")->data_type = PRIVATE_DATA;
  get_tensor_by_name(net, "inc3e")->data_type = PRIVATE_DATA;
}
