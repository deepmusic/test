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
#define MAX_NAME_LEN 64

// data_type values
#define SHARED_DATA 0  // data uses a block of shared memory
#define PRIVATE_DATA 1  // data has its own private memory
#define PARAM_DATA 2  // has own memory & pre-trained parameter is loaded
#define CPU_DATA 3 // has own CPU memory

typedef struct Tensor_
{
  char name[MAX_NAME_LEN];
  int num_items;
  int ndim;
  int shape[BATCH_SIZE][MAX_NDIM];
  int start[BATCH_SIZE];
  real* data;
  int data_type;
  int shared_block_id;
  long int max_data_size;
} Tensor;

#ifdef __cplusplus
extern "C" {
#endif

// initialize: set all values to 0
void init_tensor(Tensor* const tensor);

// set tensor's name
void set_tensor_name(Tensor* const tensor, const char* const name);

// allocate memory for tensor's data
//   allocate GPU memory in GPU mode, or CPU memory in CPU mode
//   return memory size in bytes
long int malloc_tensor_data(Tensor* const tensor,
                            real* const shared_blocks[]);

// deallocate memory
long int free_tensor_data(Tensor* const tensor);

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

// total number of elements in a tensor
long int flatten_size(const Tensor* const tensor);

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
#define MAX_NUM_LAYER_DATA 10
#define MAX_NUM_RATIOS 10
#define MAX_NUM_SCALES 10

typedef struct Net_
{
  char param_path[1024];

  Tensor tensors[MAX_NUM_TENSORS];
  int num_tensors;

  Layer layers[MAX_NUM_LAYERS];
  int num_layers;

  real* layer_data[MAX_NUM_LAYER_DATA];
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

  real anchor_ratios[MAX_NUM_RATIOS];
  real anchor_scales[MAX_NUM_SCALES];

  long int space_cpu;
  long int space;

  int initialized;

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
                      const int bias_term,
                      const int do_relu);

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
                        const int bias_term,
                        const int do_relu);

Layer* add_fc_layer(Net* const net,
                    const char* const layer_name,
                    const char* const bottom_name,
                    const char* const top_name,
                    const char* const weight_name,
                    const char* const bias_name,
                    const int num_output,
                    const int bias_term,
                    const int do_relu);

Layer* add_pool_layer(Net* const net,
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

void setup_inception(Net* const net);

void setup_frcnn(Net* const net,
                 const char* const rpn_input_name,
                 const char* const rcnn_input_name,
                 const int rpn_channels,
                 const int rpn_kernel_h, const int rpn_kernel_w,
                 const int fc6_channels, const int fc7_channels);


// --------------------------------------------------------------------------
// PVANET
// --------------------------------------------------------------------------

void construct_pvanet(Net* const net,
                      const char* const param_path);

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
                     real temp_data[]);

// channel-wise in-place softmax transform:
//   bottom[n][c][d] = exp(bottom[n][c][d]) / sum_c(exp(bottom[n][c][d]))
void softmax_inplace_forward(Tensor* const bottom3d,
                             real temp_data[]);

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
void shape_scale_const_layer(void* const net_, void* const layer_);
void shape_scale_channel_layer(void* const net_, void* const layer_);



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
//   bbox_vote: whether bounding-box voting is used (= 1) or not (= 0)
//   vote_thresh: threshold for selecting overlapped boxes
//                which are participated in bounding-box voting
void nms(const int num_boxes, real boxes[],
         int* const num_out, int keep_out[], const int base_index,
         const real nms_thresh, const int max_num_out,
         const int bbox_vote, const real vote_thresh);



// --------------------------------------------------------------------------
// transform image into network input
//   img2input
// --------------------------------------------------------------------------

void img2input(const unsigned char img[],
               Tensor* const input3d,
               Tensor* const img_info1d,
               unsigned char temp_data[],
               const int height, const int width);

void init_input_layer(Net* const net,
                      Tensor* const input3d,
                      Tensor* const img_info1d);

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
int _max_num_auxs(void);
int _max_num_ops_per_layer(void);

int _max_num_tensors(void);
int _max_num_layers(void);
int _max_num_layer_data(void);
int _max_num_ratios(void);
int _max_num_scales(void);

Net* _net(void);

void _generate_net(void);
void _init_net(void);
void _set_net_param_path(const char* const param_path);

void _add_data_layer(const char* const layer_name,
                     const char* const data_name,
                     const char* const img_info_name);

void _add_conv_layer(const char* const layer_name,
                     const char* const bottom_name,
                     const char* const top_name,
                     const char* const weight_name,
                     const char* const bias_name,
                     const int num_group, const int num_output,
                     const int kernel_h, const int kernel_w,
                     const int stride_h, const int stride_w,
                     const int pad_h, const int pad_w,
                     const int bias_term);

void _shape_net(void);
void _malloc_net(void);
void _init_layers(void);

void _release_net(void);

void _detect_net(const unsigned char image_data[],
                 const int width, const int height);

Tensor* _layer_net(const int layer_id, const int top_id);

void _print_layer(const int layer_id);

#ifdef __cplusplus
} // end extern "C"
#endif



#endif // end PVA_DL_LAYER_H
