#include "layers/operator.h"
#include <string.h>

// --------------------------------------------------------------------------
// layer-wise operator code
// --------------------------------------------------------------------------

// concat: bottom[0], bottom[1], ..., bottom[num_bottoms-1] -> top
//   bottom[m]: C_m x H x W  (C_m may different from each other)
//   top: sum(C_m) x H x W  (channel-wise concatenation)
static
void concat_forward(const Tensor* const bottom3d[],
                    Tensor* const top3d,
                    const int num_bottoms)
{
  const real* p_bottom_data[MAX_NUM_BOTTOMS];
  real* p_top_data = top3d->data;
  for (int m = 0; m < num_bottoms; ++m) {
    p_bottom_data[m] = bottom3d[m]->data;
  }

  // do forward-pass for each item in the batch
  for (int n = 0; n < bottom3d[0]->num_items; ++n) {
    // for each item, H and W should be same for all bottoms
    const int H = bottom3d[0]->shape[n][1];
    const int W = bottom3d[0]->shape[n][2];
    int top_C = 0;

    for (int m = 0; m < num_bottoms; ++m) {
      const int C = bottom3d[m]->shape[n][0];
      const int bottom_size = C * H * W;
      top_C += C;

      // channel-wise concatenation
      {
        // memcpy
      #ifdef GPU
        cudaMemcpyAsync(p_top_data, p_bottom_data[m],
                        bottom_size * sizeof(real),
                        cudaMemcpyDeviceToDevice);
      #else
        memcpy(p_top_data, p_bottom_data[m], bottom_size * sizeof(real));
      #endif

        // locate next data
        p_top_data += bottom_size;
        p_bottom_data[m] += bottom_size;
      }
    } // endfor bottoms
  } // endfor batch
}



// --------------------------------------------------------------------------
// output shape calculator code
// --------------------------------------------------------------------------

static
void concat_shape(const Tensor* const bottom3d[],
                  Tensor* const top3d,
                  const int num_bottoms)
{
  // calculate shape for each item in the batch
  for (int n = 0; n < bottom3d[0]->num_items; ++n) {
    const int H = bottom3d[0]->shape[n][1];
    const int W = bottom3d[0]->shape[n][2];
    int top_C = 0;
    for (int m = 0; m < num_bottoms; ++m) {
      top_C += bottom3d[m]->shape[n][0];
    }

    top3d->shape[n][0] = top_C;
    top3d->shape[n][1] = H;
    top3d->shape[n][2] = W;
  }

  top3d->ndim = 3;
  top3d->num_items = bottom3d[0]->num_items;
  {
    int total_size = 0;
    for (int n = 0; n < top3d->num_items; ++n) {
      top3d->start[n] = total_size;
      total_size +=
          top3d->shape[n][0] * top3d->shape[n][1] * top3d->shape[n][2];
    }
  }
}



// --------------------------------------------------------------------------
// functions for layer instance
// --------------------------------------------------------------------------

void forward_concat_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;
  concat_forward(layer->p_bottoms, get_top(layer, 0),
                 layer->num_bottoms);
}

void shape_concat_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;
  concat_shape(layer->p_bottoms, get_top(layer, 0),
               layer->num_bottoms);
}

void init_concat_layer(void* const net_, void* const layer_)
{
  return;
}

void free_concat_layer(void* const net_, void* const layer_)
{
  return;
}
