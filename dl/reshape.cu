#include "layer.h"
#include <string.h>



// --------------------------------------------------------------------------
// layer shape calculator code
// --------------------------------------------------------------------------

static
void reshape_shape(const Tensor* const bottom,
                   Tensor* const top,
                   const LayerOption* const option)
{
  for (int n = 0; n < bottom->num_items; ++n) {
    int item_size = 1, item_size_top = 1;
    int flatten_index = -1;

    for (int i = 0; i < bottom->ndim; ++i) {
      item_size *= bottom->shape[n][i];
    }

    for (int i = 0; i < option->reshape_ndim; ++i) {
      if (option->reshape[i] > 0) {
        top->shape[n][i] = option->reshape[i];
        item_size_top *= top->shape[n][i];
      }
      else if (option->reshape[i] == 0) {
        top->shape[n][i] = bottom->shape[n][i];
        item_size_top *= top->shape[n][i];
      }
      else if (flatten_index < 0) {
        flatten_index = i;
      }
      else {
        item_size_top = 0;
        flatten_index = -1;
        break;
      }
    }

    if (flatten_index >= 0) {
      top->shape[n][flatten_index] = item_size / item_size_top;
      item_size_top *= top->shape[n][flatten_index];
    }

    if (item_size_top != item_size) {
      printf("[ERROR] Wrong reshape arguments for %s -> %s: [",
             bottom->name, top->name);
      for (int i = 0; i < option->reshape_ndim - 1; ++i) {
        printf("%d, ", option->reshape[i]);
      }
      printf("%d]\n", option->reshape[option->reshape_ndim - 1]);
    }
  }

  top->ndim = option->reshape_ndim;
  top->num_items = bottom->num_items;
}



// --------------------------------------------------------------------------
// layer operator code
// --------------------------------------------------------------------------

static
void reshape_forward(const Tensor* const bottom,
                     Tensor* const top,
                     const LayerOption* const option)
{
  // copy bottom -> top, and then perform reshape operation
  if (bottom->data != top->data) {
    const long int data_size = get_data_size(bottom);

    #ifdef GPU
    cudaMemcpyAsync(top->data, bottom->data, data_size * sizeof(real),
                    cudaMemcpyDeviceToDevice);
    #else
    memcpy(top->data, bottom->data, data_size * sizeof(real));
    #endif
  }

  reshape_shape(bottom, top, option);
}



// --------------------------------------------------------------------------
// API code
// --------------------------------------------------------------------------

void forward_reshape_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;

  reshape_forward(get_bottom(layer, 0), get_top(layer, 0), &layer->option);
}

void shape_reshape_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;

  reshape_shape(get_bottom(layer, 0), get_top(layer, 0), &layer->option);
}
