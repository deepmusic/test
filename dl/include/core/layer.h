#ifndef PVA_DL_LAYER_H
#define PVA_DL_LAYER_H

#include "core/tensor.h"

// --------------------------------------------------------------------------
// definitions
// --------------------------------------------------------------------------

#define MAX_NUM_BOTTOMS 4
#define MAX_NUM_TOPS 2
#define MAX_NUM_PARAMS 2



// --------------------------------------------------------------------------
// data structures
//   LayerOption: container data structure for optional arguments
//   Layer: layer data structure
// --------------------------------------------------------------------------

typedef struct LayerOption_
{
  // for input image pre-processing operator
  int input_size;
  int unit_size;
  int max_image_size;

  // for operators using bias term
  int bias;

  // for convolution, deconvolution, fully-connected operators
  //   fully-connected operator uses out_channels and handle only
  int group;
  int num_output;
  int kernel_h, kernel_w;
  int pad_h, pad_w;
  int stride_h, stride_w;
  void* handle;

  // for RoI pooling operator
  int pooled_height;
  int pooled_width;
  real spatial_scale;
  int flatten_shape;

  // for rectified linear unit (ReLU) operator
  real negative_slope;

  // for proposal and odout operators
  real* anchor_scales;
  real* anchor_ratios;
  int num_anchor_scales;
  int num_anchor_ratios;
  int base_size;
  int feat_stride;
  int min_size;
  int pre_nms_topn;
  int post_nms_topn;
  real nms_thresh;
  real score_thresh;
  int bbox_vote;
  real vote_thresh;

  // for dropout operator
  int scaled_dropout;
  int test_dropout;
  real dropout_ratio;

  // for power operator
  real power_weight;
  real power_bias;
  real power_order;

  // for softmax operator
  int channel_axis;

  // for reshape operator
  int reshape[MAX_NDIM];
  int reshape_ndim;
} LayerOption;


typedef struct Layer_
{
  // name of this layer instance
  char name[MAX_NAME_LEN];

  // pointers to input tensors
  Tensor* p_bottoms[MAX_NUM_BOTTOMS];
  int num_bottoms;

  // pointers to output tensors
  Tensor* p_tops[MAX_NUM_TOPS];
  int num_tops;

  // pointers to pre-trained parameter tensors
  Tensor* p_params[MAX_NUM_PARAMS];
  int num_params;

  // data instance used for this layer's operator
  //   only a few operators (deconv, proposal, odout, nms) use it
  //   they define their own auxiliary data structures
  void* aux_data;

  // function pointer to this layer's operator
  //   f_forwward: forward operator
  //   f_shape: shape calculator for layer outputs
  //   f_init: layer instance initializer
  //   f_free: layer instance finalizer
  void (*f_forward)(void* net, void* layer);
  void (*f_shape)(void* net, void* layer);
  void (*f_init)(void* net, void* layer);
  void (*f_free)(void* net, void* layer);

  // a container of optional arguments
  LayerOption option;
} Layer;



// --------------------------------------------------------------------------
// functions
// --------------------------------------------------------------------------

Tensor* get_bottom(const Layer* const layer,
                   const int bottom_id);

Tensor* get_top(const Layer* const layer,
                const int top_id);

Tensor* get_param(const Layer* const layer,
                  const int param_id);

void set_bottom(Layer* const layer,
                const int bottom_id,
                Tensor* const tensor);

void set_top(Layer* const layer,
             const int top_id,
             Tensor* const tensor);

void set_param(Layer* const layer,
               const int param_id,
               Tensor* const tensor);

void add_bottom(Layer* const layer,
                Tensor* const tensor);

void add_top(Layer* const layer,
             Tensor* const tensor);

void add_param(Layer* const layer,
               Tensor* const tensor);



// --------------------------------------------------------------------------
// simple functions returning static constants
//   required for Python interface
// --------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

int _max_num_bottoms(void);

int _max_num_tops(void);

int _max_num_params(void);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end PVA_DL_LAYER_H
