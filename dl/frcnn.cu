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
  Tensor score1, score3, score5;
  Tensor bbox1, bbox3, bbox5;
  Tensor score, bbox;
  Tensor roi;
} SRPN;

typedef struct RCNN_
{
  Tensor roipool;
  Tensor fc6;
  Tensor fc7;
  Tensor score, bbox;
} RCNN;

PVANET pvanet;
SRPN srpn;
RCNN rcnn;
ConvOption conv_option1;
ConvOption conv_option2;
ConvOption conv1x1_option;
ConvOption deconv_option;
PoolOption pool_option;
ReluOption relu_option;

const Tensor* const concat_bottoms[3]
    = { &pvanet.downsample, &pvanet.conv4_3, &pvanet.upsample };

int max_layer_size = 0, max_param_size = 0;
int max_temp_size = 0, max_const_size = 0, max_tempint_size = 0;

real* layer1_data = NULL;
real* layer2_data = NULL;
real* backup1_data = NULL;
real* backup2_data = NULL;
real* backup3_data = NULL;
real* temp_data = NULL;
real* const_data = NULL;
int* tempint_data = NULL;

real* true_data = NULL;
real* input_data = NULL;
real* output_data = NULL;
real* param_data = NULL;

void load_tensor(const char* filename, Tensor* const tensor)
{
  int ndim;
  int shape[g_max_ndim];

  {
  #ifdef GPU
    int data_size = 1;
    load_data(filename, &ndim, shape, param_data);
    for (int i = 0; i < ndim; ++i) {
      data_size *= shape[i];
    }
    if (data_size != flatten_size(tensor)) {
      printf("[ERROR] Size mismatch: %s (%d) != tensor (%d)\n",
             filename, data_size, flatten_size(tensor));
    }
    cudaMemcpyAsync(tensor->data, param_data, data_size * sizeof(real),
                    cudaMemcpyHostToDevice);
  #else
    load_data(filename, &ndim, shape, tensor->data);
  #endif
  }
}

int malloc_tensor(Tensor* const tensor)
{
  const int data_size = flatten_size(tensor);

  #ifdef GPU
  cudaMalloc(&tensor->data, data_size * sizeof(real));
  #else
  tensor->data = (real*)malloc(data_size * sizeof(real));
  #endif

  return data_size;
}

void print_tensor_info(const char* name, const Tensor* const tensor)
{
  printf("%s: ", name);
  if (tensor->num_items > 1) {
    printf("%d x ", tensor->num_items);
  }
  for (int i = 0; i < tensor->ndim - 1; ++i) {
    printf("%d x ", tensor->shape[0][i]);
  }
  printf("%d\n", tensor->shape[0][tensor->ndim - 1]);
}

void forward_frcnn_7_1_1(void)
{
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
    //relu_forward_inplace(&pvanet.convf, &relu_option);
    print_tensor_info("convf", &pvanet.convf);
  }
}

void shape_frcnn_7_1_1(void)
{
  {
    int temp_size, const_size, tempint_size;

    // 1_1
    conv_option1.out_channels = 32;
    conv_shape(&pvanet.input, &pvanet.conv1_1,
               &pvanet.weight1_1, &pvanet.bias1_1,
               &temp_size, &const_size, &conv_option1);
    max_layer_size = MAX(max_layer_size, flatten_size(&pvanet.input));
    max_layer_size = MAX(max_layer_size, flatten_size(&pvanet.conv1_1));
    max_param_size = MAX(max_param_size, flatten_size(&pvanet.weight1_1));
    max_temp_size = MAX(max_temp_size, temp_size);
    max_const_size = MAX(max_const_size, const_size);
    print_tensor_info("input", &pvanet.input);
    print_tensor_info("conv1_1", &pvanet.conv1_1);

    // 1_2
    conv_option2.out_channels = 32;
    conv_shape(&pvanet.conv1_1, &pvanet.conv1_2,
               &pvanet.weight1_2, &pvanet.bias1_2,
               &temp_size, &const_size, &conv_option2);
    max_layer_size = MAX(max_layer_size, flatten_size(&pvanet.conv1_2));
    max_param_size = MAX(max_param_size, flatten_size(&pvanet.weight1_2));
    max_temp_size = MAX(max_temp_size, temp_size);
    max_const_size = MAX(max_const_size, const_size);
    print_tensor_info("conv1_2", &pvanet.conv1_2);

    // 2_1
    conv_option1.out_channels = 64;
    conv_shape(&pvanet.conv1_2, &pvanet.conv2_1,
               &pvanet.weight2_1, &pvanet.bias2_1,
               &temp_size, &const_size, &conv_option1);
    max_layer_size = MAX(max_layer_size, flatten_size(&pvanet.conv2_1));
    max_param_size = MAX(max_param_size, flatten_size(&pvanet.weight2_1));
    max_temp_size = MAX(max_temp_size, temp_size);
    max_const_size = MAX(max_const_size, const_size);
    print_tensor_info("conv2_1", &pvanet.conv2_1);

    // 2_2
    conv_option2.out_channels = 64;
    conv_shape(&pvanet.conv2_1, &pvanet.conv2_2,
               &pvanet.weight2_2, &pvanet.bias2_2,
               &temp_size, &const_size, &conv_option2);
    max_layer_size = MAX(max_layer_size, flatten_size(&pvanet.conv2_2));
    max_param_size = MAX(max_param_size, flatten_size(&pvanet.weight2_2));
    max_temp_size = MAX(max_temp_size, temp_size);
    max_const_size = MAX(max_const_size, const_size);
    print_tensor_info("conv2_2", &pvanet.conv2_2);

    // 3_1
    conv_option1.out_channels = 96;
    conv_shape(&pvanet.conv2_2, &pvanet.conv3_1,
               &pvanet.weight3_1, &pvanet.bias3_1,
               &temp_size, &const_size, &conv_option1);
    max_layer_size = MAX(max_layer_size, flatten_size(&pvanet.conv3_1));
    max_param_size = MAX(max_param_size, flatten_size(&pvanet.weight3_1));
    max_temp_size = MAX(max_temp_size, temp_size);
    max_const_size = MAX(max_const_size, const_size);
    print_tensor_info("conv3_1", &pvanet.conv3_1);

    // 3_2
    conv_option2.out_channels = 64;
    conv_shape(&pvanet.conv3_1, &pvanet.conv3_2,
               &pvanet.weight3_2, &pvanet.bias3_2,
               &temp_size, &const_size, &conv_option2);
    max_layer_size = MAX(max_layer_size, flatten_size(&pvanet.conv3_2));
    max_param_size = MAX(max_param_size, flatten_size(&pvanet.weight3_2));
    max_temp_size = MAX(max_temp_size, temp_size);
    max_const_size = MAX(max_const_size, const_size);
    print_tensor_info("conv3_2", &pvanet.conv3_2);

    // 3_3
    conv_option2.out_channels = 128;
    conv_shape(&pvanet.conv3_2, &pvanet.conv3_3,
               &pvanet.weight3_3, &pvanet.bias3_3,
               &temp_size, &const_size, &conv_option2);
    max_layer_size = MAX(max_layer_size, flatten_size(&pvanet.conv3_3));
    max_param_size = MAX(max_param_size, flatten_size(&pvanet.weight3_3));
    max_temp_size = MAX(max_temp_size, temp_size);
    max_const_size = MAX(max_const_size, const_size);
    print_tensor_info("conv3_3", &pvanet.conv3_3);

    // 4_1
    conv_option1.out_channels = 192;
    conv_shape(&pvanet.conv3_3, &pvanet.conv4_1,
               &pvanet.weight4_1, &pvanet.bias4_1,
               &temp_size, &const_size, &conv_option1);
    max_layer_size = MAX(max_layer_size, flatten_size(&pvanet.conv4_1));
    max_param_size = MAX(max_param_size, flatten_size(&pvanet.weight4_1));
    max_temp_size = MAX(max_temp_size, temp_size);
    max_const_size = MAX(max_const_size, const_size);
    print_tensor_info("conv4_1", &pvanet.conv4_1);

    // 4_2
    conv_option2.out_channels = 128;
    conv_shape(&pvanet.conv4_1, &pvanet.conv4_2,
               &pvanet.weight4_2, &pvanet.bias4_2,
               &temp_size, &const_size, &conv_option2);
    max_layer_size = MAX(max_layer_size, flatten_size(&pvanet.conv4_2));
    max_param_size = MAX(max_param_size, flatten_size(&pvanet.weight4_2));
    max_temp_size = MAX(max_temp_size, temp_size);
    max_const_size = MAX(max_const_size, const_size);
    print_tensor_info("conv4_2", &pvanet.conv4_2);

    // 4_3
    conv_option2.out_channels = 256;
    conv_shape(&pvanet.conv4_2, &pvanet.conv4_3,
               &pvanet.weight4_3, &pvanet.bias4_3,
               &temp_size, &const_size, &conv_option2);
    max_layer_size = MAX(max_layer_size, flatten_size(&pvanet.conv4_3));
    max_param_size = MAX(max_param_size, flatten_size(&pvanet.weight4_3));
    max_temp_size = MAX(max_temp_size, temp_size);
    max_const_size = MAX(max_const_size, const_size);
    print_tensor_info("conv4_3", &pvanet.conv4_3);

    // 5_1
    conv_option1.out_channels = 384;
    conv_shape(&pvanet.conv4_3, &pvanet.conv5_1,
               &pvanet.weight5_1, &pvanet.bias5_1,
               &temp_size, &const_size, &conv_option1);
    max_layer_size = MAX(max_layer_size, flatten_size(&pvanet.conv5_1));
    max_param_size = MAX(max_param_size, flatten_size(&pvanet.weight5_1));
    max_temp_size = MAX(max_temp_size, temp_size);
    max_const_size = MAX(max_const_size, const_size);
    print_tensor_info("conv5_1", &pvanet.conv5_1);

    // 5_2
    conv_option2.out_channels = 256;
    conv_shape(&pvanet.conv5_1, &pvanet.conv5_2,
               &pvanet.weight5_2, &pvanet.bias5_2,
               &temp_size, &const_size, &conv_option2);
    max_layer_size = MAX(max_layer_size, flatten_size(&pvanet.conv5_2));
    max_param_size = MAX(max_param_size, flatten_size(&pvanet.weight5_2));
    max_temp_size = MAX(max_temp_size, temp_size);
    max_const_size = MAX(max_const_size, const_size);
    print_tensor_info("conv5_2", &pvanet.conv5_2);

    // 5_3
    conv_option2.out_channels = 512;
    conv_shape(&pvanet.conv5_2, &pvanet.conv5_3,
               &pvanet.weight5_3, &pvanet.bias5_3,
               &temp_size, &const_size, &conv_option2);
    max_layer_size = MAX(max_layer_size, flatten_size(&pvanet.conv5_3));
    max_param_size = MAX(max_param_size, flatten_size(&pvanet.weight5_3));
    max_temp_size = MAX(max_temp_size, temp_size);
    max_const_size = MAX(max_const_size, const_size);
    print_tensor_info("conv5_3", &pvanet.conv5_3);

    // downsample
    pool_shape(&pvanet.conv3_3, &pvanet.downsample,
               &tempint_size, &pool_option);
    max_layer_size = MAX(max_layer_size, flatten_size(&pvanet.downsample));
    max_tempint_size = MAX(max_tempint_size, tempint_size);
    print_tensor_info("downsample", &pvanet.downsample);

    // upsample
    deconv_shape(&pvanet.conv5_3, &pvanet.upsample,
                 &pvanet.weight_up, &pvanet.bias_up,
                 &temp_size, &const_size, &deconv_option);
    max_layer_size = MAX(max_layer_size, flatten_size(&pvanet.upsample));
    max_param_size = MAX(max_param_size, flatten_size(&pvanet.weight_up));
    max_temp_size = MAX(max_temp_size, temp_size);
    max_const_size = MAX(max_const_size, const_size);
    print_tensor_info("upsample", &pvanet.upsample);

    // concat
    concat_shape(concat_bottoms, &pvanet.concat, 3);
    max_layer_size = MAX(max_layer_size, flatten_size(&pvanet.concat));
    print_tensor_info("concat", &pvanet.concat);

    // convf
    conv1x1_option.out_channels = 512;
    conv_shape(&pvanet.concat, &pvanet.convf,
               &pvanet.weightf, &pvanet.biasf,
               &temp_size, &const_size, &conv1x1_option);
    max_layer_size = MAX(max_layer_size, flatten_size(&pvanet.convf));
    max_param_size = MAX(max_param_size, flatten_size(&pvanet.weightf));
    max_temp_size = MAX(max_temp_size, temp_size);
    max_const_size = MAX(max_const_size, const_size);
    print_tensor_info("convf", &pvanet.convf);
  }
}

void init_frcnn_7_1_1(void)
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
  }

  // calculate maximum size
  pvanet.input.num_items = 1;
  pvanet.input.ndim = 3;
  pvanet.input.shape[0][0] = 3;
  pvanet.input.shape[0][1] = 640;
  pvanet.input.shape[0][2] = 1024;
  shape_frcnn_7_1_1();

  // memory allocation
  {
    long int space = 0;

    // space for data loading
    {
      input_data = (real*)malloc(flatten_size(&pvanet.input) * sizeof(real));
      output_data = (real*)malloc(max_layer_size * sizeof(real));
      true_data = (real*)malloc(max_layer_size * sizeof(real));
      param_data = (real*)malloc(max_param_size * sizeof(real));
      space += flatten_size(&pvanet.input) + max_layer_size * 2
               + max_param_size;
    }

    // space required for forward-pass
    {
    #ifdef GPU
      cudaMalloc(&layer1_data, max_layer_size * sizeof(real));
      cudaMalloc(&layer2_data, max_layer_size * sizeof(real));
      cudaMalloc(&temp_data, max_temp_size * sizeof(real));
      cudaMalloc(&const_data, max_const_size * sizeof(real));
      cudaMalloc(&tempint_data, max_tempint_size * sizeof(int));
      cudaMalloc(&backup1_data, max_layer_size * sizeof(real));
      cudaMalloc(&backup2_data, max_layer_size * sizeof(real));
      cudaMalloc(&backup3_data, max_layer_size * sizeof(real));
    #else
      layer1_data = (real*)malloc(max_layer_size * sizeof(real));
      layer2_data = (real*)malloc(max_layer_size * sizeof(real));
      temp_data = (real*)malloc(max_temp_size * sizeof(real));
      const_data = (real*)malloc(max_const_size * sizeof(real));
      tempint_data = (int*)malloc(max_tempint_size * sizeof(int));
      backup1_data = (real*)malloc(max_layer_size * sizeof(real));
      backup2_data = (real*)malloc(max_layer_size * sizeof(real));
      backup3_data = (real*)malloc(max_layer_size * sizeof(real));
    #endif
      space += max_layer_size * 4 + max_temp_size + max_const_size
               + max_tempint_size;
    }

    // parameters
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
      space += malloc_tensor(&pvanet.bias_up);
      space += malloc_tensor(&pvanet.weightf);
      space += malloc_tensor(&pvanet.biasf);
    }

    space = DIV_THEN_CEIL(space * sizeof(real),  1000000);
    printf("%ldMB memory allocated\n", space);
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

  // parameter loading
  {
    load_tensor("../data/temp/conv1_1_param0.bin", &pvanet.weight1_1);
    load_tensor("../data/temp/conv1_1_param1.bin", &pvanet.bias1_1);
    load_tensor("../data/temp/conv1_2_param0.bin", &pvanet.weight1_2);
    load_tensor("../data/temp/conv1_2_param1.bin", &pvanet.bias1_2);
    load_tensor("../data/temp/conv2_1_param0.bin", &pvanet.weight2_1);
    load_tensor("../data/temp/conv2_1_param1.bin", &pvanet.bias2_1);
    load_tensor("../data/temp/conv2_2_param0.bin", &pvanet.weight2_2);
    load_tensor("../data/temp/conv2_2_param1.bin", &pvanet.bias2_2);
    load_tensor("../data/temp/conv3_1_param0.bin", &pvanet.weight3_1);
    load_tensor("../data/temp/conv3_1_param1.bin", &pvanet.bias3_1);
    load_tensor("../data/temp/conv3_2_param0.bin", &pvanet.weight3_2);
    load_tensor("../data/temp/conv3_2_param1.bin", &pvanet.bias3_2);
    load_tensor("../data/temp/conv3_3_param0.bin", &pvanet.weight3_3);
    load_tensor("../data/temp/conv3_3_param1.bin", &pvanet.bias3_3);
    load_tensor("../data/temp/conv4_1_param0.bin", &pvanet.weight4_1);
    load_tensor("../data/temp/conv4_1_param1.bin", &pvanet.bias4_1);
    load_tensor("../data/temp/conv4_2_param0.bin", &pvanet.weight4_2);
    load_tensor("../data/temp/conv4_2_param1.bin", &pvanet.bias4_2);
    load_tensor("../data/temp/conv4_3_param0.bin", &pvanet.weight4_3);
    load_tensor("../data/temp/conv4_3_param1.bin", &pvanet.bias4_3);
    load_tensor("../data/temp/conv5_1_param0.bin", &pvanet.weight5_1);
    load_tensor("../data/temp/conv5_1_param1.bin", &pvanet.bias5_1);
    load_tensor("../data/temp/conv5_2_param0.bin", &pvanet.weight5_2);
    load_tensor("../data/temp/conv5_2_param1.bin", &pvanet.bias5_2);
    load_tensor("../data/temp/conv5_3_param0.bin", &pvanet.weight5_3);
    load_tensor("../data/temp/conv5_3_param1.bin", &pvanet.bias5_3);
    load_tensor("../data/temp/upsample_param0.bin", &pvanet.weight_up);
    load_tensor("../data/temp/convf_param0.bin", &pvanet.weightf);
    load_tensor("../data/temp/convf_param1.bin", &pvanet.biasf);
  }
}

int main(int argc, char* argv[])
{
  // CUDA initialization
  #ifdef GPU
  {
    printf("set device\n");
    cudaSetDevice(0);
    conv_option1.handle = (cublasHandle_t*)malloc(sizeof(cublasHandle_t));
    if (cublasCreate((cublasHandle_t*)conv_option1.handle)
          != CUBLAS_STATUS_SUCCESS) {
      printf("cublas creation failed\n");
    }
    conv_option2.handle = conv_option1.handle;
  }
  #endif

  // PVANET initialization
  init_frcnn_7_1_1();

  // input data loading
  {
    int ndim;
    int shape[g_max_ndim];
    int input_size;

    load_data("../data/temp/conv1_1_bottom0.bin", &ndim, shape, input_data);
    pvanet.input.num_items = shape[0];
    pvanet.input.ndim = ndim - 1;
    input_size = 0;
    for (int n = 0; n < pvanet.input.num_items; ++n) {
      int size_n = 1;
      for (int i = 0; i < pvanet.input.ndim; ++i) {
        pvanet.input.shape[n][i] = shape[i + 1];
        size_n *= shape[i + 1];
      }
      pvanet.input.start[n] = input_size;
      input_size += size_n;
    }

  #ifdef GPU
    cudaMemcpyAsync(layer1_data, input_data, input_size * sizeof(real),
                    cudaMemcpyHostToDevice);
  #else
    memcpy(layer1_data, input_data, input_size * sizeof(real));
  #endif

    print_tensor_info("input data loaded", &pvanet.input);
  }

  // network reshape
  shape_frcnn_7_1_1();

  // forward-pass
  printf("forward-pass start\n");
  forward_frcnn_7_1_1();
  printf("forward-pass end\n");

  // retrieve output
  {
    const int output_size = flatten_size(&pvanet.convf);

  #ifdef GPU
    cudaMemcpyAsync(output_data, pvanet.convf.data,
                    output_size * sizeof(real),
                    cudaMemcpyDeviceToHost);
  #else
    memcpy(output_data, pvanet.convf.data, output_size * sizeof(real));
  #endif
  }

  // verify results
  {
    const int output_size = flatten_size(&pvanet.convf);

    int ndim;
    int shape[g_max_ndim];
    load_data("../data/temp/convf_top0.bin", &ndim, shape, true_data);

    for (int i = 0; i < output_size; ++i) {
      real diff = ABS(true_data[i] - output_data[i]);
      diff /= 1e-10f + MIN(ABS(true_data[i]),  ABS(output_data[i]));
      #ifdef GPU
      if (diff > 0) {
        printf("%d: %.6f %.6f\n", i, true_data[i], output_data[i]);
      }
      #else
      if (diff > 1e-3f) {
        printf("%d: %.6f %.6f\n", i, true_data[i], output_data[i]);
      }
      #endif
    }
  }

  // memory deallocation
  {
    if (true_data) free(true_data);
    if (input_data) free(input_data);
    if (output_data) free(output_data);
    if (param_data) free(param_data);
  }
  #ifdef GPU
  {
    if (layer1_data) cudaFree(layer1_data);
    if (layer2_data) cudaFree(layer2_data);
    if (backup1_data) cudaFree(backup1_data);
    if (backup2_data) cudaFree(backup2_data);
    if (backup3_data) cudaFree(backup3_data);
    if (temp_data) cudaFree(temp_data);
    if (tempint_data) cudaFree(tempint_data);
    if (const_data) cudaFree(const_data);

    if (cublasDestroy(*((cublasHandle_t*)conv_option1.handle))
        != CUBLAS_STATUS_SUCCESS) {
      printf("cublas destruction failed\n");
    }

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
  }
  #else
  {
    if (layer1_data) free(layer1_data);
    if (layer2_data) free(layer2_data);
    if (backup1_data) free(backup1_data);
    if (backup2_data) free(backup2_data);
    if (backup3_data) free(backup3_data);
    if (temp_data) free(temp_data);
    if (tempint_data) free(tempint_data);
    if (const_data) free(const_data);

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
  }
  #endif

  return 0;
}
