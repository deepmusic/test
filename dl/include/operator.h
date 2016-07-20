// --------------------------------------------------------------------------
// outer interface for all layer-wise operators
//   forward_XXX_layer: forward operator
//   shape_XXX_layer: shape calculator for output tensors
//   malloc_XXX_layer: auxiliary data initializer (optional)
//   free_XXX_layer: auxiliary data finalizer (optional)
// --------------------------------------------------------------------------

#ifndef PVA_DL_OPERATOR_H
#define PVA_DL_OPERATOR_H

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

#endif // end PVA_DL_OPERATOR_H
