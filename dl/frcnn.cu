#include "layer.h"
#include <stdio.h>
#include <string.h>

typedef struct Layer_
{
  char name[32];
  Tensor** p_bottoms;
  int num_bottoms;
  Tensor* tops;
  int num_tops;
  int* allocate_top_data;
  Tensor* params;
  int num_params;
  LayerOption option;
} Layer;

long int malloc_layer(Layer* const layer)
{
  long int space_cpu = 0;

  layer->p_bottoms = (layer->num_bottoms <= 0) ? NULL :
                     (Tensor**)malloc(layer->num_bottoms * sizeof(Tensor*));
  space_cpu += layer->num_bottoms * sizeof(Tensor*);

  layer->tops = (layer->num_tops <= 0) ? NULL :
                (Tensor*)malloc(layer->num_tops * sizeof(Tensor));
  layer->allocate_top_data = (layer->num_tops <= 0) ? NULL:
                             (int*)calloc(layer->num_tops, sizeof(int));
  space_cpu += layer->num_tops * (sizeof(Tensor) + sizeof(int));

  layer->params = (layer->num_params <= 0) ? NULL : 
                  (Tensor*)malloc(layer->num_params * sizeof(Tensor));
  space_cpu += layer->num_params * sizeof(Tensor);

  return space_cpu;
}

long int malloc_load_layer_data(Layer* const layer,
                                const char* const name,
                                real* const temp_cpu_space)
{
  long int space = 0;

  for (int i = 0; i < layer->num_tops; ++i) {
    if (layer->allocate_top_data[i]) {
      space += malloc_tensor(&layer->tops[i]);
    }
  }

  for (int i = 0; i < layer->num_params; ++i) {
    char path[1024];
    printf("malloc param %d\n", i);
    space += malloc_tensor(&layer->params[i]);
    sprintf(path, "params/%s_param%d.bin", name, i);
    printf("load param %s\n", path);
    load_tensor(path, &layer->params[i], temp_cpu_space);
  }

  return space;
}

void free_layer(Layer* const layer)
{
  if (layer->p_bottoms) {
    free(layer->p_bottoms);
    layer->p_bottoms = NULL;
  }

  if (layer->tops) {
    for (int i = 0; i < layer->num_tops; ++i) {
      if (layer->allocate_top_data[i]) {
        #ifdef GPU
        cudaFree(layer->tops[i].data);
        layer->tops[i].data = NULL;
        #else
        free(layer->tops[i].data);
        layer->tops[i].data = NULL;
        #endif
      }
    }
    free(layer->tops);
    layer->tops = NULL;
    free(layer->allocate_top_data);
    layer->allocate_top_data = NULL;
  }

  if (layer->params) {
    for (int i = 0; i < layer->num_params; ++i) {
      #ifdef GPU
      cudaFree(layer->params[i].data);
      layer->params[i].data = NULL;
      #else
      free(layer->params[i].data);
      layer->params[i].data = NULL;
      #endif
    }
    free(layer->params);
    layer->params = NULL;
  }

  free(layer);
  layer->params = NULL;
}

#define MAX_NUM_LAYERS 100
#define MAX_NUM_LAYER_DATA 5
#define MAX_NUM_RATIOS 10
#define MAX_NUM_SCALES 10

typedef struct Net_
{
  Layer* layers[MAX_NUM_LAYERS];
  int num_layers;

  real* layer_data[MAX_NUM_LAYER_DATA];
  int reserved_layer_data[MAX_NUM_LAYER_DATA];
  real* input_cpu_data;
  real* output_cpu_data;
  int layer_size;
  int num_layer_data;

  real* param_cpu_data;
  int param_size;

  real* temp_data;
  real* temp_cpu_data;
  int temp_size;

  int* tempint_data;
  int* tempint_cpu_data;
  int tempint_size;

  real* const_data;
  int const_size;

  Tensor* img_info;
  real* anchors;
  real anchor_ratios[MAX_NUM_RATIOS];
  real anchor_scales[MAX_NUM_SCALES];

  long int space_cpu;
  long int space;

  #ifdef GPU
  cublasHandle_t cublas_handle;
  #endif
} Net;

void malloc_net(Net* const net)
{
  long int space_cpu = 0;
  long int space = 0;

  for (int i = 0; i < net->num_layer_data; ++i) {
    #ifdef GPU
    cudaMalloc(&net->layer_data[i], net->layer_size * sizeof(real));
    #else
    net->layer_data[i] = (real*)malloc(net->layer_size * sizeof(real));
    #endif
    net->reserved_layer_data[i] = 0;
  }
  space += net->num_layer_data * net->layer_size * sizeof(real);

  #ifdef GPU
  {
    cudaMalloc(&net->temp_data, net->temp_size * sizeof(real));
    cudaMalloc(&net->tempint_data, net->tempint_size * sizeof(int));
    cudaMalloc(&net->const_data, net->const_size * sizeof(real));
  }
  #else
  {
    net->temp_data = (real*)malloc(net->temp_size * sizeof(real));
    net->tempint_data = (int*)malloc(net->tempint_size * sizeof(int));
    net->const_data = (real*)malloc(net->const_size * sizeof(real));
  }
  #endif
  space += sizeof(real) * (net->temp_size + net->const_size)
           + sizeof(int) * (net->tempint_size);

  net->input_cpu_data = (real*)malloc(net->layer_size * sizeof(real));
  net->output_cpu_data = (real*)malloc(net->layer_size * sizeof(real));
  net->param_cpu_data = (real*)malloc(net->param_size * sizeof(real));
  net->temp_cpu_data = (real*)malloc(net->temp_size * sizeof(real));
  net->tempint_cpu_data = (int*)malloc(net->tempint_size * sizeof(int));
  space_cpu += sizeof(real) * (2 * net->layer_size + net->param_size
                               + net->temp_size)
               + sizeof(int) * (net->tempint_size);

  // data initialization
  {
  #ifdef GPU
    for (int i = 0; i < net->const_size; ++i) {
      net->output_cpu_data[i] = 1;
    }
    cudaMemcpy(net->const_data, net->output_cpu_data,
               net->const_size * sizeof(real),
               cudaMemcpyHostToDevice);
  #else
    for (int i = 0; i < net->const_size; ++i) {
      net->const_data[i] = 1;
    }
  #endif
  }

  for (int i = 0; i < net->num_layers; ++i) {
    space += malloc_load_layer_data(net->layers[i], net->layers[i]->name,
                                    net->param_cpu_data);
  }

  net->img_info->data
      = (real*)malloc(flatten_size(net->img_info) * sizeof(real));
  space_cpu += sizeof(real) * flatten_size(net->img_info);

  // acquire CuBLAS handle
  #ifdef GPU
  {
    if (cublasCreate(&net->cublas_handle) != CUBLAS_STATUS_SUCCESS) {
      printf("cublas creation failed\n");
    }
  }
  #endif

  net->space_cpu = space_cpu;
  net->space = space;
}

void free_net(Net* const net)
{
  for (int i = 0; i < net->num_layers; ++i) {
    free_layer(net->layers[i]);
  }

  for (int i = 0; i < net->num_layer_data; ++i) {
    #ifdef GPU
    cudaFree(net->layer_data[i]);
    #else
    free(net->layer_data[i]);
    #endif
    net->layer_data[i] = NULL;
  }

  #ifdef GPU
  {
    cudaFree(net->temp_data);
    cudaFree(net->tempint_data);
    cudaFree(net->const_data);
    cudaFree(net->anchors);
  }
  #else
  {
    free(net->temp_data);
    free(net->tempint_data);
    free(net->const_data);
    free(net->anchors);
  }
  #endif

  free(net->input_cpu_data);
  free(net->output_cpu_data);
  free(net->param_cpu_data);
  free(net->temp_cpu_data);
  free(net->tempint_cpu_data);

  free(net->img_info->data);
  free(net->img_info);

  net->temp_data = NULL;
  net->tempint_data = NULL;
  net->const_data = NULL;
  net->input_cpu_data = NULL;
  net->output_cpu_data = NULL;
  net->param_cpu_data = NULL;
  net->temp_cpu_data = NULL;
  net->tempint_cpu_data = NULL;
  net->anchors = NULL;
  net->img_info = NULL;

  #ifdef GPU
  {
    if (cublasDestroy(net->cublas_handle) != CUBLAS_STATUS_SUCCESS) {
      printf("cublas destruction failed\n");
    }
  }
  #endif
}

void update_net_size(Net* const net,
                     const int bottom_size,
                     const int top_size,
                     const int param_size,
                     const int temp_size,
                     const int tempint_size,
                     const int const_size)
{
  net->layer_size = MAX(net->layer_size,  bottom_size);
  net->layer_size = MAX(net->layer_size,  top_size);
  net->param_size = MAX(net->param_size,  param_size);
  net->temp_size = MAX(net->temp_size,  temp_size);
  net->tempint_size = MAX(net->tempint_size,  tempint_size);
  net->const_size = MAX(net->const_size,  const_size);
}

real* get_layer_data(Net* const net)
{
  for (int i = 0; i < net->num_layer_data; ++i) {
    if (!net->reserved_layer_data[i]) {
      net->reserved_layer_data[i] = 1;
      return net->layer_data[i];
    }
  }

  printf("[ERROR] Not enough temporary space for storing layer output!\n");
  return NULL;
}

void shape_conv_layer(Net* const net, Layer* const layer)
{
  int temp_size, const_size;
  Tensor* p_bias = (layer->option.bias) ? &layer->params[1] : NULL;

  conv_shape(layer->p_bottoms[0], &layer->tops[0],
             &layer->params[0], p_bias,
             &temp_size, &const_size, &layer->option);

  update_net_size(net,
      flatten_size(layer->p_bottoms[0]),  flatten_size(&layer->tops[0]),
      flatten_size(&layer->params[0]),  temp_size,  0,  const_size);
}

void shape_deconv_layer(Net* const net, Layer* const layer)
{
  int temp_size, const_size;
  Tensor* p_bias = (layer->option.bias) ? &layer->params[1] : NULL;

  deconv_shape(layer->p_bottoms[0], &layer->tops[0],
               &layer->params[0], p_bias,
               &temp_size, &const_size, &layer->option);

  update_net_size(net,
      flatten_size(layer->p_bottoms[0]),  flatten_size(&layer->tops[0]),
      flatten_size(&layer->params[0]),  temp_size,  0,  const_size);
}

void shape_pool_layer(Net* const net, Layer* const layer)
{
  int tempint_size;

  pool_shape(layer->p_bottoms[0], &layer->tops[0],
             &tempint_size, &layer->option);

  update_net_size(net,
      flatten_size(layer->p_bottoms[0]),  flatten_size(&layer->tops[0]),
      0,  0,  tempint_size,  0);
}

void shape_concat_layer(Net* const net, Layer* const layer)
{
  concat_shape(layer->p_bottoms, &layer->tops[0],
               &layer->option);

  update_net_size(net,
      0,  flatten_size(&layer->tops[0]),
      0,  0,  0,  0);
}

void shape_proposal_layer(Net* const net, Layer* const layer)
{
  int temp_size, tempint_size;

  proposal_shape(layer->p_bottoms[0], &layer->tops[0],
                 &temp_size, &tempint_size, &layer->option);

  update_net_size(net,
      0,  0,
      0,  temp_size,  tempint_size,  0);
}

void forward_conv_relu_layer(Net* const net, Layer* const layer)
{
  Tensor* p_bias = (layer->option.bias) ? &layer->params[1] : NULL;

  conv_forward(layer->p_bottoms[0], &layer->tops[0],
               &layer->params[0], p_bias,
               net->temp_data, net->const_data, &layer->option);
  relu_forward_inplace(&layer->tops[0], &layer->option);
  print_tensor_info(layer->name, &layer->tops[0]);
}

void forward_deconv_layer(Net* const net, Layer* const layer)
{
  Tensor* p_bias = (layer->option.bias) ? &layer->params[1] : NULL;

  deconv_forward(layer->p_bottoms[0], &layer->tops[0],
                 &layer->params[0], p_bias,
                 net->temp_data, net->const_data, &layer->option);
  print_tensor_info(layer->name, &layer->tops[0]);
}

void forward_pool_layer(Net* const net, Layer* const layer)
{
  pool_forward(layer->p_bottoms[0], &layer->tops[0],
               net->tempint_data, &layer->option);
  print_tensor_info(layer->name, &layer->tops[0]);
}

void forward_concat_layer(Net* const net, Layer* const layer)
{
  concat_forward(layer->p_bottoms, &layer->tops[0],
                 &layer->option);
  print_tensor_info(layer->name, &layer->tops[0]);
}

void forward_proposal_layer(Net* const net, Layer* const layer)
{
  proposal_forward(layer->p_bottoms[0], layer->p_bottoms[1], net->img_info,
                   &layer->tops[0], net->anchors,
                   net->temp_cpu_data, net->tempint_cpu_data,
                   net->temp_data, net->tempint_data,
                   &layer->option);
  print_tensor_info("img_info", net->img_info);
  print_tensor_info(layer->name, &layer->tops[0]);
}

void forward_frcnn_7_1_1(Net* net)
{
  // PVANET
  {
    // data
    net->layers[0]->tops[0].data = net->layer_data[0];

    // 1_1, 1_2, 2_1, 2_2, 3_1, 3_2, 3_3
    for (int i = 1; i <= 7; ++i) {
      net->layers[i]->tops[0].data = net->layer_data[i % 2];
      forward_conv_relu_layer(net, net->layers[i]);
    }

    // downsample
    net->layers[8]->tops[0].data = net->layer_data[2];
    forward_pool_layer(net, net->layers[8]);

    // 4_1, 4_2
    for (int i = 9; i <= 10; ++i) {
      net->layers[i]->tops[0].data = net->layer_data[i % 2];
      forward_conv_relu_layer(net, net->layers[i]);
    }

    // 4_3
    net->layers[11]->tops[0].data = net->layer_data[3];
    forward_conv_relu_layer(net, net->layers[11]);

    // 5_1, 5_2, 5_3
    for (int i = 12; i <= 14; ++i) {
      net->layers[i]->tops[0].data = net->layer_data[i % 2];
      forward_conv_relu_layer(net, net->layers[i]);
    }

    // upsample
    net->layers[15]->tops[0].data = net->layer_data[4];
    forward_deconv_layer(net, net->layers[15]);

    // concat
    net->layers[16]->tops[0].data = net->layer_data[0];
    forward_concat_layer(net, net->layers[16]);

    // convf
    net->layers[17]->tops[0].data = net->layer_data[4];
    forward_conv_relu_layer(net, net->layers[17]);
  }

  // Multi-scale RPN
  {
    // rpn_1, 3, 5
    for (int i = 18; i <= 26; i += 3) {
      // rpn_conv1, 3, 5
      net->layers[i]->tops[0].data = net->layer_data[0];
      forward_conv_relu_layer(net, net->layers[i]);

      // rpn_cls_score1, 3, 5
      forward_conv_relu_layer(net, net->layers[i + 1]);

      // rpn_bbox_pred1, 3, 5
      forward_conv_relu_layer(net, net->layers[i + 2]);
    }

    // rpn_score
    net->layers[27]->tops[0].data = net->layer_data[0];
    forward_concat_layer(net, net->layers[27]);

    // pred
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
    pred->data = score->data;
    softmax_inplace_forward(pred, net->temp_data);
    print_tensor_info(net->layers[28]->name, pred);

    // pred reshape
    pred->ndim = 4;
    pred->num_items = score->num_items;
    for (int n = 0; n < score->num_items; ++n) {
      pred->shape[n][0] = 2;
      pred->shape[n][1] = score->shape[n][0] / 2;
      pred->shape[n][2] = score->shape[n][1];
      pred->shape[n][3] = score->shape[n][2];
    }
    print_tensor_info("rpn_pred_reshape", pred);

    // rpn_bbox
    net->layers[29]->tops[0].data = net->layer_data[0];
    forward_concat_layer(net, net->layers[29]);

    // bbox reshape
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

    // proposal
    forward_proposal_layer(net, net->layers[30]);
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
    "fc6_1", "fc6_2", "fc7_1", "fc7_2",
    "cls_score", "cls_pred", "bbox_pred",
    "out"
  };

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
      net->layers[i]->option.handle = (void*)&net->cublas_handle;
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
      net->layers[i]->option.handle = (void*)&net->cublas_handle;
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
  }

  {
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

    net->layers[30]->num_bottoms = 2;
    net->layers[30]->num_tops = 1;
    net->layers[30]->num_params = 0;

    for (int i = 31; i < net->num_layers; ++i) {
      net->layers[i]->num_bottoms = 0;
      net->layers[i]->num_tops = 0;
      net->layers[i]->num_params = 0;
    }
  }

  {
    net->num_layers = 41;
    net->num_layer_data = 5;
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
    // rpn_bbox
    shape_concat_layer(net, net->layers[29]);
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
    // img_info
    net->img_info->ndim = 1;
    net->img_info->num_items = bbox->num_items;
    for (int n = 0; n < net->img_info->num_items; ++n) {
      net->img_info->shape[n][0] = 4;
      net->img_info->start[n] = n * 4;
    }
    // proposal
    shape_proposal_layer(net, net->layers[30]);
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

  printf("Max layer size = %d\n", net->layer_size);
  printf("Max param size = %d\n", net->param_size);
  printf("Max temp size = %d\n", net->temp_size);
  printf("Max tempint size = %d\n", net->tempint_size);
  printf("Max const size = %d\n", net->const_size);

  malloc_net(net);

  net->space_cpu += space_cpu;

  const int num_anchors = net->layers[30]->option.num_scales
                          * net->layers[30]->option.num_ratios
                          * net->layers[30]->option.num_concats;
  #ifdef GPU
  {
    cudaMalloc(&net->anchors, num_anchors * 4 * sizeof(real));
    generate_anchors(net->param_cpu_data, &net->layers[30]->option);
    cudaMemcpy(net->anchors, net->param_cpu_data,
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
                   const char* const filename[], const int num_images)
{
  Tensor* input = &net->layers[0]->tops[0];
  input->data = net->input_cpu_data;
  input->ndim = 3;
  input->num_items = 0;
  input->start[0] = 0;

  net->img_info->ndim = 1;
  net->img_info->num_items = 0;

  for (int i = 0; i < num_images; ++i) {
    load_image(filename[i], input, net->img_info);
  }

  #ifdef GPU
  cudaMemcpyAsync(net->layer_data[0], net->input_cpu_data,
                  flatten_size(input) * sizeof(real),
                  cudaMemcpyHostToDevice);
  #else
  memcpy(net->layer_data[0], net->input_cpu_data,
         flatten_size(input) * sizeof(real));
  #endif

  // network reshape
  shape_frcnn_7_1_1(net);

  print_tensor_info("input data loaded", input);
}

void get_output(Net* net, const int image_start_index, FILE* fp)
{
  // retrieve output
  {
    const Tensor* const out = &net->layers[net->num_layers - 1]->tops[0];
    const int output_size = flatten_size(out);

  #ifdef GPU
    cudaMemcpyAsync(net->output_cpu_data, out->data,
                    output_size * sizeof(real),
                    cudaMemcpyDeviceToHost);
  #else
    memcpy(net->output_data, out->data, output_size * sizeof(real));
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
        prepare_input(&frcnn, (const char * const *)&line, count);

        // forward-pass
        printf("forward-pass start\n");
        forward_frcnn_7_1_1(&frcnn);
        printf("forward-pass end\n");

        // retrieve output & save to file
        //get_output(&frcnn, total_count, fp_out);
        cudaMemcpy(frcnn.output_cpu_data, frcnn.layers[30]->tops[0].data,
                   flatten_size(&frcnn.layers[30]->tops[0]),
                   cudaMemcpyDeviceToHost);
        for (int n = 0; n < frcnn.layers[30]->tops[0].num_items; ++n) {
          printf("Image %d, start = %d\n", n, frcnn.layers[30]->tops[0].start[n]);
          real* roi = &frcnn.output_cpu_data[frcnn.layers[30]->tops[0].start[n]];
          for (int r = 0; r < frcnn.layers[30]->tops[0].shape[n][0]; ++r) {
            printf("  Box %d:  %.2f %.2f %.2f %.2f\n",
                   r, roi[r * 4 + 0], roi[r * 4 + 1], roi[r * 4 + 2], roi[r * 4 + 3]);
          }
        }

        total_count += count;
        count = 0;
      }
    }

    if (count > 0) {
      prepare_input(&frcnn, (const char * const *)&line, count);
      printf("forward-pass start\n");
      forward_frcnn_7_1_1(&frcnn);
      printf("forward-pass end\n");
      //get_output(&frcnn, total_count, fp_out);
    }

    fclose(fp_list);
    fclose(fp_out);
  }


  // end
  free_net(&frcnn);

  return 0;
}
