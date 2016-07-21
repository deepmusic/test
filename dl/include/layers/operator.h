// --------------------------------------------------------------------------
// outer interface for all layer-wise operators
//   forward_XXX_layer: forward operator
//   shape_XXX_layer: shape calculator for output tensors
//   init_XXX_layer: layer instance initializer
//   free_XXX_layer: layer instance finalizer
// --------------------------------------------------------------------------

#ifndef PVA_DL_OPERATOR_H
#define PVA_DL_OPERATOR_H

#include "core/net.h"

// --------------------------------------------------------------------------
// input image pre-processing
// --------------------------------------------------------------------------

void forward_image_layer(void* const net_, void* const layer_);
void shape_image_layer(void* const net_, void* const layer_);
void init_image_layer(void* const net_, void* const layer_);
void free_image_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// convolution
// --------------------------------------------------------------------------

void forward_conv_layer(void* const net_, void* const layer_);
void shape_conv_layer(void* const net_, void* const layer_);
void init_conv_layer(void* const net_, void* const layer_);
void free_conv_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// deconvolution
// --------------------------------------------------------------------------

void forward_deconv_layer(void* const net_, void* const layer_);
void shape_deconv_layer(void* const net_, void* const layer_);
void init_deconv_layer(void* const net_, void* const layer_);
void free_deconv_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// fully-connected
// --------------------------------------------------------------------------

void forward_fc_layer(void* const net_, void* const layer_);
void shape_fc_layer(void* const net_, void* const layer_);
void init_fc_layer(void* const net_, void* const layer_);
void free_fc_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// pooling
// --------------------------------------------------------------------------

void forward_pool_layer(void* const net_, void* const layer_);
void shape_pool_layer(void* const net_, void* const layer_);
void init_pool_layer(void* const net_, void* const layer_);
void free_pool_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// RoI pooling
// --------------------------------------------------------------------------

void forward_roipool_layer(void* const net_, void* const layer_);
void shape_roipool_layer(void* const net_, void* const layer_);
void init_roipool_layer(void* const net_, void* const layer_);
void free_roipool_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// top-n proposal generation
// --------------------------------------------------------------------------

void forward_proposal_layer(void* const net_, void* const layer_);
void shape_proposal_layer(void* const net_, void* const layer_);
void init_proposal_layer(void* const net_, void* const layer_);
void free_proposal_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// concat
// --------------------------------------------------------------------------

void forward_concat_layer(void* const net_, void* const layer_);
void shape_concat_layer(void* const net_, void* const layer_);
void init_concat_layer(void* const net_, void* const layer_);
void free_concat_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// object detection output
// --------------------------------------------------------------------------

void forward_odout_layer(void* const net_, void* const layer_);
void shape_odout_layer(void* const net_, void* const layer_);
void init_odout_layer(void* const net_, void* const layer_);
void free_odout_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// softmax
// --------------------------------------------------------------------------

void forward_softmax_layer(void* const net_, void* const layer_);
void shape_softmax_layer(void* const net_, void* const layer_);
void init_softmax_layer(void* const net_, void* const layer_);
void free_softmax_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// dropout
// --------------------------------------------------------------------------

void forward_dropout_layer(void* const net_, void* const layer_);
void shape_dropout_layer(void* const net_, void* const layer_);
void init_dropout_layer(void* const net_, void* const layer_);
void free_dropout_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// ReLU
// --------------------------------------------------------------------------

void forward_relu_layer(void* const net_, void* const layer_);
void shape_relu_layer(void* const net_, void* const layer_);
void init_relu_layer(void* const net_, void* const layer_);
void free_relu_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// scale
// --------------------------------------------------------------------------

void forward_scale_layer(void* const net_, void* const layer_);
void shape_scale_layer(void* const net_, void* const layer_);
void init_scale_layer(void* const net_, void* const layer_);
void free_scale_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// power
// --------------------------------------------------------------------------

void forward_power_layer(void* const net_, void* const layer_);
void shape_power_layer(void* const net_, void* const layer_);
void init_power_layer(void* const net_, void* const layer_);
void free_power_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// eltwise operations
// --------------------------------------------------------------------------

void forward_eltwise_layer(void* const net_, void* const layer_);
void shape_eltwise_layer(void* const net_, void* const layer_);
void init_eltwise_layer(void* const net_, void* const layer_);
void free_eltwise_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// reshape
// --------------------------------------------------------------------------

void forward_reshape_layer(void* const net_, void* const layer_);
void shape_reshape_layer(void* const net_, void* const layer_);
void init_reshape_layer(void* const net_, void* const layer_);
void free_reshape_layer(void* const net_, void* const layer_);



// --------------------------------------------------------------------------
// Concatenated ReLU
// --------------------------------------------------------------------------

void forward_crelu_layer(void* const net_, void* const layer_);
void shape_crelu_layer(void* const net_, void* const layer_);
void init_crelu_layer(void* const net_, void* const layer_);
void free_crelu_layer(void* const net_, void* const layer_);

#endif // end PVA_DL_OPERATOR_H
