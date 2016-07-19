#ifndef PVA_DL_LAYER_H
#define PVA_DL_LAYER_H

#define DEBUG
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
// --------------------------------------------------------------------------

typedef float real;
#define MAX_NDIM 5
#define MAX_NAME_LEN 64

// data_type values
#define SHARED_DATA 0  // data uses a block of shared memory
#define PRIVATE_DATA 1  // data has its own private memory
#define PARAM_DATA 2  // has own memory & pre-trained parameter is loaded

typedef struct Tensor_
{
  char name[MAX_NAME_LEN];
  int num_items;
  int ndim;
  int shape[BATCH_SIZE][MAX_NDIM];
  int start[BATCH_SIZE];
  real* data;
  int data_type;
} Tensor;

#ifdef __cplusplus
extern "C" {
#endif

// initialize: set all values to 0
void init_tensor(Tensor* const tensor);

// set tensor's name
void set_tensor_name(Tensor* const tensor, const char* const name);

// total number of elements in a tensor
long int get_data_size(const Tensor* const tensor);

// allocate memory for tensor's data
//   allocate GPU memory in GPU mode, or CPU memory in CPU mode
//   return memory size in bytes
long int malloc_tensor_data(Tensor* const tensor,
                            real* const shared_block);

// deallocate memory
void free_tensor_data(Tensor* const tensor);

// load binary data from file & store to CPU memory
//   cpu_data: pointer to CPU memory for storing data
void load_from_binary_file(const char* const filename,
                           int* const ndim,
                           int shape[],
                           real cpu_data[]);

// load binary data from file & copy to memory where tensor data refers
//   temp_cpu_data: pointer to CPU memory for loading data temporarily
void load_tensor_data(const char* const filename,
                      Tensor* const tensor,
                      real temp_cpu_data[]);

// save tensor data to binary file
//   temp_cpu_data: pointer to CPU memory for storing data temporarily
//                  can be set NULL if tensor data refers to CPU memory
void save_tensor_data(const char* const filename,
                      const Tensor* const tensor,
                      real temp_cpu_data[]);

// print shapes for all batch items in tensor
void print_tensor_info(const Tensor* const tensor);

#ifdef __cplusplus
} //end extern "C"
#endif



// --------------------------------------------------------------------------
// layer data structure & some functions
// --------------------------------------------------------------------------

#define MAX_NUM_BOTTOMS 4
#define MAX_NUM_TOPS 2
#define MAX_NUM_PARAMS 2

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
  int base_size;
  int feat_stride;
  int min_size;
  int pre_nms_topn;
  int post_nms_topn;
  real nms_thresh;
  real score_thresh;
  int bbox_vote;
  real vote_thresh;

  int scaled;
  int test;
  real threshold;

  real scale_weight;
  real scale_bias;

  int num_bottoms;

  int channel_axis;

  int reshape[MAX_NDIM];
  int reshape_ndim;
} LayerOption;

typedef struct Layer_
{
  char name[MAX_NAME_LEN];

  Tensor* p_bottoms[MAX_NUM_BOTTOMS];
  int num_bottoms;

  Tensor* p_tops[MAX_NUM_TOPS];
  int num_tops;

  Tensor* p_params[MAX_NUM_PARAMS];
  int num_params;

  void* aux_data;

  void (*f_forward)(void*, void*);
  void (*f_shape)(void*, void*);
  void (*f_free)(void*, void*);

  LayerOption option;
} Layer;

#ifdef __cplusplus
extern "C" {
#endif

void init_layer(Layer* const layer);
void set_layer_name(Layer* const layer, const char* const name);

void set_bottom(Layer* const layer, const int bottom_id,
                Tensor* const tensor);
void set_top(Layer* const layer, const int top_id,
             Tensor* const tensor);
void set_param(Layer* const layer, const int param_id,
               Tensor* const tensor);

void add_bottom(Layer* const layer, Tensor* const tensor);
void add_top(Layer* const layer, Tensor* const tensor);
void add_param(Layer* const layer, Tensor* const tensor);

Tensor* get_bottom(const Layer* const layer, const int bottom_id);
Tensor* get_top(const Layer* const layer, const int top_id);
Tensor* get_param(const Layer* const layer, const int param_id);

#ifdef __cplusplus
} // end extern "C"
#endif



// --------------------------------------------------------------------------
// net data structure & some functions
// --------------------------------------------------------------------------

#define MAX_NUM_TENSORS 400
#define MAX_NUM_LAYERS 400
#define MAX_NUM_SHARED_BLOCKS 10

typedef struct Net_
{
  char param_path[1024];

  Tensor tensors[MAX_NUM_TENSORS];
  int num_tensors;

  Layer layers[MAX_NUM_LAYERS];
  int num_layers;

  real* p_shared_blocks[MAX_NUM_SHARED_BLOCKS];
  int num_shared_blocks;

  int num_output_boxes;

  real* temp_data;
  real* temp_cpu_data;
  long int temp_space;

  real* const_data;
  long int const_space;

  long int space_cpu;
  long int space;

  int initialized;

  int input_scale;

  #ifdef GPU
  cublasHandle_t blas_handle;
  #else
  int blas_handle;
  #endif
} Net;

#ifdef __cplusplus
extern "C" {
#endif

Tensor* get_tensor(Net* const net, const int tensor_id);
Layer* get_layer(Net* const net, const int layer_id);

int get_tensor_id(Net* const net, const Tensor* const tensor);
int get_layer_id(Net* const net, const Layer* const layer);

Tensor* find_tensor_by_name(Net* const net, const char* const name);
Layer* find_layer_by_name(Net* const net, const char* const name);

Tensor* add_tensor(Net* const net, const char* const name);
Layer* add_layer(Net* const net, const char* const name);

Tensor* find_or_add_tensor(Net* const net, const char* const name);
Layer* find_or_add_layer(Net* const net, const char* const name);

Tensor* get_tensor_by_name(Net* const net, const char* const name);
Layer* get_layer_by_name(Net* const net, const char* const name);

void init_net(Net* const net);

void malloc_net(Net* const net);

void free_net(Net* const net);

void forward_net(Net* const net);

void shape_net(Net* const net);

void assign_shared_blocks(Net* const net);

void update_temp_space(Net* const net, const long int space);
void update_const_space(Net* const net, const long int space);

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
// Network generation
// --------------------------------------------------------------------------

Layer* add_data_layer(Net* const net,
                      const char* const layer_name,
                      const char* const data_name,
                      const char* const img_info_name);

Layer* add_conv_layer(Net* const net,
                      const char* const layer_name,
                      const char* const bottom_name,
                      const char* const top_name,
                      const char* const weight_name,
                      const char* const bias_name,
                      const int num_group, const int num_output,
                      const int kernel_h, const int kernel_w,
                      const int stride_h, const int stride_w,
                      const int pad_h, const int pad_w,
                      const int bias_term);

Layer* add_deconv_layer(Net* const net,
                        const char* const layer_name,
                        const char* const bottom_name,
                        const char* const top_name,
                        const char* const weight_name,
                        const char* const bias_name,
                        const int num_group, const int num_output,
                        const int kernel_h, const int kernel_w,
                        const int stride_h, const int stride_w,
                        const int pad_h, const int pad_w,
                        const int bias_term);

Layer* add_fc_layer(Net* const net,
                    const char* const layer_name,
                    const char* const bottom_name,
                    const char* const top_name,
                    const char* const weight_name,
                    const char* const bias_name,
                    const int num_output,
                    const int bias_term);

Layer* add_max_pool_layer(Net* const net,
                          const char* const layer_name,
                          const char* const bottom_name,
                          const char* const top_name,
                          const int kernel_h, const int kernel_w,
                          const int stride_h, const int stride_w,
                          const int pad_h, const int pad_w);

Layer* add_scale_const_layer(Net* const net,
                             const char* const layer_name,
                             const char* const bottom_name,
                             const char* const top_name,
                             const real weight,
                             const real bias,
                             const int bias_term);

Layer* add_scale_channel_layer(Net* const net,
                               const char* const layer_name,
                               const char* const bottom_name,
                               const char* const top_name,
                               const char* const weight_name,
                               const char* const bias_name,
                               const int bias_term);

Layer* add_concat_layer(Net* const net,
                        const char* const layer_name,
                        const char* const p_bottom_names[],
                        const char* const top_name,
                        const int num_bottoms);

Layer* add_eltwise_layer(Net* const net,
                         const char* const layer_name,
                         const char* const p_bottom_names[],
                         const char* const top_name,
                         const int num_bottoms);

Layer* add_relu_layer(Net* const net,
                      const char* const layer_name,
                      const char* const bottom_name,
                      const char* const top_name,
                      const real negative_slope);

Layer* add_dropout_layer(Net* const net,
                         const char* const layer_name,
                         const char* const bottom_name,
                         const char* const top_name,
                         const int is_test,
                         const int is_scaled);

Layer* add_reshape_layer(Net* const net,
                         const char* const layer_name,
                         const char* const bottom_name,
                         const char* const top_name,
                         const int shape[],
                         const int ndim);

Layer* add_softmax_layer(Net* const net,
                         const char* const layer_name,
                         const char* const bottom_name,
                         const char* const top_name,
                         const int channel_axis);

void setup_shared_cnn(Net* const net);
void setup_shared_cnn_light(Net* const net);

void setup_frcnn(Net* const net,
                 const char* const rpn_input_name,
                 const char* const rcnn_input_name,
                 const int rpn_channels,
                 const int rpn_kernel_h, const int rpn_kernel_w,
                 const int fc_compress,
                 const int fc6_channels, const int fc7_channels,
                 const int pre_nms_topn, const int post_nms_topn);


// --------------------------------------------------------------------------
// PVANET
// --------------------------------------------------------------------------

void construct_pvanet(Net* const net,
                      const char* const param_path,
                      const int is_light_model,
                      const int fc_compress,
                      const int pre_nms_topn,
                      const int post_nms_topn,
                      const int input_scale);

void set_input_pvanet(Net* const net,
                      const unsigned char* const images_data[],
                      const int heights[],
                      const int widths[],
                      const int num_images);

void get_output_pvanet(Net* const net,
                       const int image_start_index,
                       FILE* fp);

void process_pvanet(Net* const net,
                    const unsigned char image_data[],
                    const int height, const int width,
                    FILE* fp);

void process_batch_pvanet(Net* const net,
                          const unsigned char* const images_data[],
                          const int heights[],
                          const int widths[],
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
void malloc_deconv_layer(void* const net_, void* const layer_);
void free_deconv_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// fully-connected
// --------------------------------------------------------------------------

void forward_fc_layer(void* const net_, void* const layer_);
void shape_fc_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// pooling
// --------------------------------------------------------------------------

void forward_max_pool_layer(void* const net_, void* const layer_);
void shape_pool_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// RoI pooling
// --------------------------------------------------------------------------

void forward_roipool_layer(void* const net_, void* const layer_);
void shape_roipool_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// top-n proposal generation
// --------------------------------------------------------------------------

void forward_proposal_layer(void* const net_, void* const layer_);
void shape_proposal_layer(void* const net_, void* const layer_);
void malloc_proposal_layer(void* const net_, void* const layer_);
void free_proposal_layer(void* const net_, void* const layer_);



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
void malloc_odout_layer(void* const net_, void* const layer_);
void free_odout_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// softmax
// --------------------------------------------------------------------------

void forward_softmax_layer(void* const net_, void* const layer_);
void shape_softmax_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// dropout
// --------------------------------------------------------------------------

void forward_dropout_layer(void* const net_, void* const layer_);
void shape_dropout_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// ReLU
// --------------------------------------------------------------------------

void forward_relu_layer(void* const net_, void* const layer_);
void shape_relu_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// scale
// --------------------------------------------------------------------------

void forward_scale_const_layer(void* const net_, void* const layer_);
void forward_scale_channel_layer(void* const net_, void* const layer_);
void shape_scale_const_layer(void* const net_, void* const layer_);
void shape_scale_channel_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// eltwise operations
// --------------------------------------------------------------------------

void forward_eltwise_sum_layer(void* const net_, void* const layer_);
void shape_eltwise_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// reshape
// --------------------------------------------------------------------------

void forward_reshape_layer(void* const net_, void* const layer_);
void shape_reshape_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// transform image into network input
//   img2input
// --------------------------------------------------------------------------

void img2input(const unsigned char img[],
               Tensor* const input3d,
               Tensor* const img_info1d,
               unsigned char temp_data[],
               const int height, const int width,
               const int input_scale);

void init_input_layer(Net* const net,
                      Tensor* const input3d,
                      Tensor* const img_info1d);



// --------------------------------------------------------------------------
// common functions
//   iou
//   soft_box
//   nms
// --------------------------------------------------------------------------

// "IoU = intersection area / union area" of two boxes A, B
//   A, B: 4-dim array (x1, y1, x2, y2)
real iou(const real A[], const real B[]);

// quick-sort a list of boxes in descending order of their scores (CPU)
//   list_cpu: num_boxes x 5 array,  (x1, y1, x2, y2, score) for each box
//             located at main memory
//   if num_top <= end,  only top-k results are guaranteed to be sorted
//   (for efficient computation)
void sort_box(real list_cpu[], const int start, const int end,
              const int num_top);

// given box proposals (sorted in descending order of their scores),
// discard a box if it is significantly overlapped with
// one or more previous (= scored higher) boxes
//   num_boxes: number of box proposals given
//   boxes: "num_boxes x 5" array (x1, y1, x2, y2, score)
//          sorted in descending order of scores
//   aux_data: auxiliary data for NMS operation
//   num_out: number of remaining boxes
//   index_out_cpu: "num_out x 1" array
//                  indices of remaining boxes
//                  allocated at main memory
//   base_index: a constant added to index_out_cpu,  usually 0
//               index_out_cpu[i] = base_index + actual index in boxes
//   nms_thresh: threshold for determining "significant overlap"
//               if "intersection area / union area > nms_thresh",
//               two boxes are thought of as significantly overlapped
//   bbox_vote: whether bounding-box voting is used (= 1) or not (= 0)
//   vote_thresh: threshold for selecting overlapped boxes
//                which are participated in bounding-box voting
void nms(const int num_boxes, real boxes[], void* const aux_data,
         int* const num_out, int index_out_cpu[],
         const int base_index, const real nms_thresh, const int max_num_out,
         const int bbox_vote, const real vote_thresh);

void malloc_nms_aux_data(void** const p_aux_data,
                         int num_boxes,
                         long int* const p_space_cpu,
                         long int* const p_space_gpu);

void free_nms_aux_data(void* const aux_data);

#ifdef __cplusplus
} // end extern "C"
#endif



// --------------------------------------------------------------------------
// shared library interface
// --------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

int _batch_size_net(void);
int _max_ndim(void);
int _max_name_len(void);

int _max_num_bottoms(void);
int _max_num_tops(void);
int _max_num_params(void);

int _max_num_tensors(void);
int _max_num_layers(void);
int _max_num_shared_blocks(void);

Net* _net(void);

void _init_net(void);
void _set_net_param_path(const char* const param_path);

void _shape_net(void);
void _malloc_net(void);

void _release_net(void);

void _detect_net(const unsigned char image_data[],
                 const int width, const int height);

Tensor* _layer_net(const int layer_id, const int top_id);

#ifdef __cplusplus
} // end extern "C"
#endif



#endif // end PVA_DL_LAYER_H
