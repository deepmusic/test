#ifndef PVA_DL_LAYER_H
#define PVA_DL_LAYER_H

// --------------------------------------------------------------------------
// include cuda & blas library
// --------------------------------------------------------------------------

#ifdef GPU
  #include "cuda_settings.h"
#else
  #include <cblas.h>
#endif

#include <stdlib.h>



// --------------------------------------------------------------------------
// simple math operations
// --------------------------------------------------------------------------

#define ABS(x)  ((x) > 0 ? (x) : (-(x)))
#define DIV_THEN_CEIL(x, y)  (((x) + (y) - 1) / (y))
#define ROUND(x)  ((int)((x) + 0.5f))

#ifdef GPU
  #define MIN(x, y)  min(x, y)
  #define MAX(x, y)  max(x, y)
#else
  #define MIN(x, y)  ((x) < (y) ? (x) : (y))
  #define MAX(x, y)  ((x) > (y) ? (x) : (y))
#endif



// --------------------------------------------------------------------------
// tensor data structure & some functions
//   struct Tensor
//   flatten_size
//   print_tensor_info
//   malloc_tensor
//   load_data
//   load_tensor
// --------------------------------------------------------------------------

typedef float real;
#define g_max_num_items 128
#define g_max_ndim 5

typedef struct Tensor_
{
  int num_items;
  int ndim;
  int shape[g_max_num_items][g_max_ndim];
  int start[g_max_num_items];
  real* data;
} Tensor;

// initialize: set all values to 0
void init_tensor(Tensor* const tensor);

// allocate memory for tensor
//   allocate GPU memory in GPU mode, or CPU memory in CPU mode
//   return memory size in bytes
long int malloc_tensor_data(Tensor* const tensor);

// deallocate memory & set all values to 0
void free_tensor_data(Tensor* const tensor);

// load binary data from file & store to CPU memory
//   data: pointer to CPU memory for storing data
//         if NULL, allocate new memory & load data & return pointer
real* load_data(const char* const filename,
                int* const ndim,
                int* const shape,
                real* data);

// load binary data from file & copy to memory where tensor occupies
//   temp_data: pointer to CPU memory for loading data temporarily
//              not used (i.e., can be NULL) if tensor occupies CPU memory
void load_tensor(const char* const filename,
                 Tensor* const tensor,
                 real* const temp_data);

// save tensor data to binary file
//   temp_data: pointer to CPU memory for storing data temporarily
//              not used (i.e., can be NULL) if tensor occupies CPU memory
void save_tensor_data(const char* const filename,
                      const Tensor* const tensor,
                      real* const temp_data);

// total number of elements in a tensor
long int flatten_size(const Tensor* const tensor);

// print shapes for all batch items in tensor
void print_tensor_info(const char* name,
                       const Tensor* const tensor);



// --------------------------------------------------------------------------
// load image & transform into network input
//   load_image
//   img2input
// --------------------------------------------------------------------------

void load_image(const char* const filename,
                Tensor* const input3d,
                Tensor* const img_info1d,
                real* const temp_data);

void img2input(const unsigned char* const img,
               Tensor* const input3d,
               Tensor* const img_info1d,
               unsigned char* const temp_data,
               const int height, const int width, const int stride);



// --------------------------------------------------------------------------
// layer data structure & some functions
// --------------------------------------------------------------------------

# define MAX_NUM_OPS_PER_LAYER 5

// layer options
typedef struct LayerOption_
{
  int num_groups;
  int out_channels;
  int kernel_h, kernel_w;
  int pad_h, pad_w;
  int stride_h, stride_w;
  int bias;
  void* handle;

  int pooled_height;
  int pooled_width;
  real spatial_scale;
  int flatten;

  real negative_slope;

  real* scales;
  real* ratios;
  int num_scales;
  int num_ratios;
  int num_concats;
  int base_size;
  int feat_stride;
  int min_size;
  int pre_nms_topn;
  int post_nms_topn;
  real nms_thresh;
  real score_thresh;

  int scaled;
  int test;
  real threshold;
} LayerOption;

typedef struct Layer_
{
  char name[32];

  Tensor** p_bottoms;
  int num_bottoms;

  Tensor* tops;
  int* allocate_top_data;
  int num_tops;

  Tensor* params;
  int num_params;

  real** p_aux_data;
  int num_aux_data;

  void (*f_forward[MAX_NUM_OPS_PER_LAYER])(void*, void*);
  void (*f_shape[MAX_NUM_OPS_PER_LAYER])(void*, void*);
  void (*f_init[MAX_NUM_OPS_PER_LAYER])(void*, void*);
  LayerOption option;
} Layer;

void init_layer(Layer* const layer);

long int malloc_layer(Layer* const layer);



// --------------------------------------------------------------------------
// net data structure & some functions
// --------------------------------------------------------------------------

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
  long int layer_size;
  int num_layer_data;

  real* param_cpu_data;
  long int param_size;

  real* temp_data;
  real* temp_cpu_data;
  long int temp_size;

  int* tempint_data;
  int* tempint_cpu_data;
  long int tempint_size;

  real* const_data;
  long int const_size;

  Tensor* img_info;
  real anchor_ratios[MAX_NUM_RATIOS];
  real anchor_scales[MAX_NUM_SCALES];

  long int space_cpu;
  long int space;

  int initialized;

  #ifdef GPU
  cublasHandle_t cublas_handle;
  #endif
} Net;

void init_net(Net* const net);

void malloc_net(Net* const net);

void free_net(Net* const net);

void init_layers(Net* const net);

void forward_net(Net* const net);

void shape_net(Net* const net);

void update_net_size(Net* const net,
                     const Layer* const layer,
                     const int temp_size,
                     const int tempint_size,
                     const int const_size);

void save_layer_tops(void* const net_, void* const layer_);

void print_layer_tops(const Net* const net,
                      const Layer* const layer);



// --------------------------------------------------------------------------
// PVANET 7.1.1
// --------------------------------------------------------------------------

void construct_frcnn_7_1_1(Net* net);



// --------------------------------------------------------------------------
// convolution
// --------------------------------------------------------------------------

void forward_conv_layer(void* const net, void* const layer);
void shape_conv_layer(void* const net, void* const layer);



// --------------------------------------------------------------------------
// deconvolution
// --------------------------------------------------------------------------

void forward_deconv_layer(void* const net_, void* const layer_);
void shape_deconv_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// fully-connected
// --------------------------------------------------------------------------

void forward_fc_layer(void* const net_, void* const layer_);
void shape_fc_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// pooling
// --------------------------------------------------------------------------

void forward_pool_layer(void* const net_, void* const layer_);
void shape_pool_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// RoI pooling
// --------------------------------------------------------------------------

void forward_roipool_layer(void* const net_, void* const layer_);
void shape_roipool_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// top-n proposal generation
// --------------------------------------------------------------------------

void init_proposal_layer(void* const net_, void* const layer_);
void forward_proposal_layer(void* const net_, void* const layer_);
void shape_proposal_layer(void* const net_, void* const layer_);

// given a base box, enumerate transformed boxes of varying sizes and ratios
//   option->base_size: base box's width & height (i.e., base box is square)
//   option->scales: "option->num_scales x 1" array
//                   varying scale factor for base box
//   option->ratios: "option->num_ratios x 1" array
//                   varying height-width ratio
//   option->num_concats: repeat count of anchor set generation
//                        (required for separated RPN)
//   anchors: "num_boxes x 4" array,  (x1, y1, x2, y2) for each box
//     num_boxes = total number of transformations
//         = option->num_scales * option->num_ratios * option->num_concats
void generate_anchors(real* const anchors,
                      const LayerOption* const option);



// --------------------------------------------------------------------------
// concat
// --------------------------------------------------------------------------

void forward_concat_layer(void* const net_, void* const layer_);
void shape_concat_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// object detection output
// --------------------------------------------------------------------------

void forward_odout_layer(void* const net_, void* const layer_);
void shape_odout_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// object detection test output (for measuring MAP performance)
// --------------------------------------------------------------------------

void forward_odtest_layer(void* const net_, void* const layer_);
void shape_odtest_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// softmax
//   softmax_forward
//   softmax_inplace_forward
//   softmax_shape
// --------------------------------------------------------------------------

void forward_rpn_pred_layer(void* const net_, void* const layer_);
void shape_rpn_pred_layer(void* const net_, void* const layer_);

void forward_rcnn_pred_layer(void* const net_, void* const layer_);
void shape_rcnn_pred_layer(void* const net_, void* const layer_);

void forward_rpn_bbox_layer(void* const net_, void* const layer_);
void shape_rpn_bbox_layer(void* const net_, void* const layer_);

void forward_rcnn_bbox_layer(void* const net_, void* const layer_);
void shape_rcnn_bbox_layer(void* const net_, void* const layer_);

// channel-wise softmax transform: bottom3d (N x C x D) -> top3d (N x C x D)
//   top[n][c][d] = exp(bottom[n][c][d]) / sum_c(exp(bottom[n][c][d]))
//   temp_data: N * D array,  temporary space for channel-wise sum or max
//     e.g., temp_data[n][d] = sum_c(exp(bottom[n][c][d]))
void softmax_forward(const Tensor* const bottom3d,
                     Tensor* const top3d,
                     real* const temp_data);

// channel-wise in-place softmax transform:
//   bottom[n][c][d] = exp(bottom[n][c][d]) / sum_c(exp(bottom[n][c][d]))
void softmax_inplace_forward(Tensor* const bottom3d,
                             real* const temp_data);

void softmax_shape(const Tensor* const bottom3d,
                   Tensor* const top3d,
                   int* const temp_size);



// --------------------------------------------------------------------------
// dropout
// --------------------------------------------------------------------------

void forward_dropout_layer(void* const net_, void* const layer_);
void forward_inplace_dropout_layer(void* const net_, void* const layer_);
void shape_dropout_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// ReLU
// --------------------------------------------------------------------------

void forward_relu_layer(void* const net_, void* const layer_);
void forward_inplace_relu_layer(void* const net_, void* const layer_);
void shape_relu_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// NMS operation
//   nms
// --------------------------------------------------------------------------

// given box proposals (sorted in descending order of their scores),
// discard a box if it is significantly overlapped with
// one or more previous (= scored higher) boxes
//   num_boxes: number of box proposals given
//   boxes: "num_boxes x 5" array (x1, y1, x2, y2, score)
//          sorted in descending order of scores
//   num_out: number of remaining boxes
//   keep_out: "num_out x 1" array
//             indices of remaining boxes
//   base_index: a constant added to keep_out,  usually 0
//               keep_out[i] = base_index + actual index in boxes
//   nms_thresh: threshold for determining "significant overlap"
//               if "intersection area / union area > nms_thresh",
//               two boxes are thought of as significantly overlapped
void nms(const int num_boxes, const real* const boxes,
         int* const num_out, int* const keep_out, const int base_index,
         const real nms_thresh, const int max_num_out);


#endif // endifndef PVA_DL_LAYER_H
