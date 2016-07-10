#ifndef PVA_DL_LAYER_H
#define PVA_DL_LAYER_H

//#define DEBUG
//#define MKL
#define DEMO

#ifdef DEMO
  #define BATCH_SIZE 1
#else
  #define BATCH_SIZE 4
#endif

// --------------------------------------------------------------------------
// include cuda & blas library
// --------------------------------------------------------------------------

#ifdef GPU
  #include <cublas_v2.h>
  #include <cuda.h>
  #include <cuda_runtime.h>
  #include <curand.h>
  #include <driver_types.h>
#elif defined(MKL)
  #include <mkl_cblas.h>
  #include <math.h>
#else
  #include <cblas.h>
  #include <math.h>
#endif

#include <stdio.h>
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
#define MAX_NDIM 5

typedef struct Tensor_
{
  char name[32];
  int num_items;
  int ndim;
  int shape[BATCH_SIZE][MAX_NDIM];
  int start[BATCH_SIZE];
  real* data;
  void* alive_until;
  int has_own_memory;
  int data_id;
  long int max_data_size;
} Tensor;

#ifdef __cplusplus
extern "C" {
#endif

// initialize: set all values to 0
void init_tensor(Tensor* const tensor);

// allocate memory for tensor
//   allocate GPU memory in GPU mode, or CPU memory in CPU mode
//   return memory size in bytes
long int malloc_tensor_data(Tensor* const tensor);

// deallocate memory
long int free_tensor_data(Tensor* const tensor);

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
void print_tensor_info(const char* const name,
                       const Tensor* const tensor);

#ifdef __cplusplus
} //end extern "C"
#endif



// --------------------------------------------------------------------------
// layer data structure & some functions
// --------------------------------------------------------------------------

#define MAX_NUM_BOTTOMS 4
#define MAX_NUM_TOPS 1
#define MAX_NUM_PARAMS 2
#define MAX_NUM_AUXS 1
#define MAX_NUM_OPS_PER_LAYER 5

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

  real scale_weight;
  real scale_bias;

  int num_bottoms;
} LayerOption;

typedef struct Layer_
{
  char name[32];

  Tensor* p_bottoms[MAX_NUM_BOTTOMS];
  int num_bottoms;

  Tensor tops[MAX_NUM_TOPS];
  int num_tops;

  Tensor params[MAX_NUM_PARAMS];
  int num_params;

  real* p_aux_data[MAX_NUM_AUXS];
  int num_aux_data;

  void (*f_forward[MAX_NUM_OPS_PER_LAYER])(void*, void*);
  void (*f_shape[MAX_NUM_OPS_PER_LAYER])(void*, void*);
  void (*f_init[MAX_NUM_OPS_PER_LAYER])(void*, void*);
  LayerOption option;
} Layer;

#ifdef __cplusplus
extern "C" {
#endif

void init_layer(Layer* const layer);

void set_layer_name(Layer* const layer, const char* const name);

#ifdef __cplusplus
} // end extern "C"
#endif



// --------------------------------------------------------------------------
// net data structure & some functions
// --------------------------------------------------------------------------

#define MAX_NUM_LAYERS 300
#define MAX_NUM_LAYER_DATA 6
#define MAX_NUM_RATIOS 10
#define MAX_NUM_SCALES 10

typedef struct Net_
{
  char param_path[1024];

  Layer layers[MAX_NUM_LAYERS];
  int num_layers;

  real* layer_data[MAX_NUM_LAYER_DATA];
  void* reserved_until[MAX_NUM_LAYER_DATA];
  long int layer_size;
  int num_layer_data;

  int num_output_boxes;

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

  Tensor img_info;
  real anchor_ratios[MAX_NUM_RATIOS];
  real anchor_scales[MAX_NUM_SCALES];

  long int space_cpu;
  long int space;

  int initialized;

  #ifdef GPU
  cublasHandle_t cublas_handle;
  #endif
} Net;

#ifdef __cplusplus
extern "C" {
#endif

long int malloc_top_data(Net* const net, Layer* const layer,
                         const int top_id);

long int free_top_data(Net* const net, Layer* const layer,
                       const int top_id);

void init_net(Net* const net);

void malloc_net(Net* const net);

void free_net(Net* const net);

void init_layers(Net* const net);

void forward_net(Net* const net);

void shape_net(Net* const net);

void assign_layer_data(Net* const net, Tensor* const tensor);

void deallocate_layer_data(Net* const net, Tensor* const tensor);

void update_net_size(Net* const net,
                     const Layer* const layer,
                     const int temp_size,
                     const int tempint_size,
                     const int const_size);

void save_layer_tops(void* const net_, void* const layer_);

void print_layer_tops(void* const net_, void* const layer_);

#ifdef __cplusplus
} // end extern "C"
#endif



#ifdef __cplusplus
extern "C" {
#endif

// --------------------------------------------------------------------------
// PVANET
// --------------------------------------------------------------------------

void construct_pvanet(Net* const net,
                      const char* const param_path);

void set_input_pvanet(Net* const net,
                      const unsigned char* const * const images_data,
                      const int* const heights,
                      const int* const widths,
                      const int num_images);

void get_output_pvanet(Net* const net,
                       const int image_start_index,
                       FILE* fp);

void process_pvanet(Net* const net,
                    const unsigned char* const image_data,
                    const int height, const int width,
                    FILE* fp);

void process_batch_pvanet(Net* const net,
                          const unsigned char* const * const images_data,
                          const int* const heights,
                          const int* const widths,
                          const int num_images,
                          FILE* fp);



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
// CReLU
// --------------------------------------------------------------------------

void forward_crelu_layer(void* const net_, void* const layer_);
void shape_crelu_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// scale
// --------------------------------------------------------------------------

void forward_scale_const_layer(void* const net_, void* const layer_);
void forward_scale_channel_layer(void* const net_, void* const layer_);
void forward_inplace_scale_const_layer(void* const net_,
                                       void* const layer_);
void forward_inplace_scale_channel_layer(void* const net_,
                                         void* const layer_);
void shape_scale_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// eltwise operations
// --------------------------------------------------------------------------

void forward_eltwise_sum_layer(void* const net_, void* const layer_);
void shape_eltwise_layer(void* const net_, void* const layer_);



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

#ifdef __cplusplus
} // end extern "C"
#endif



// --------------------------------------------------------------------------
// transform image into network input
//   img2input
// --------------------------------------------------------------------------

void img2input(const unsigned char* const img,
               Tensor* const input3d,
               Tensor* const img_info1d,
               unsigned char* const temp_data,
               const int height, const int width);

void input_init_shape(Net* const net,
                      Tensor* const input3d,
                      Tensor* const img_info1d);



// --------------------------------------------------------------------------
// shared library interface
// --------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

int _batch_size_net(void);

void _init_net(void);

void _release_net(void);

void _detect_net(const unsigned char* const image_dta, 
                 const int width, const int height);

Tensor* _layer_net(const int layer_id, const int top_id);

void _print_layer(const int layer_id);

#ifdef __cplusplus
} // end extern "C"
#endif



#endif // end PVA_DL_LAYER_H
