#include "layer.h"
#include <string.h>

#include <time.h>

#include "logger.h"

static float a_time[8] = { 0, };
static clock_t tick0, tick1;

// --------------------------------------------------------------------------
// layer operator code
//   concat_forward
// --------------------------------------------------------------------------

// concat: bottom[0], bottom[1], ..., bottom[M-1] -> top
//   M = option->num_concats
//   bottom[m]: C_m x H x W  (C_m may different from each other)
//   top: sum(C_m) x H x W  (channel-wise concatenation)
void concat_forward(const Tensor* const bottom3d[],
                    Tensor* const top3d,
                    const LayerOption* const option)
{
  tick0 = clock();

  const int num_bottoms = option->num_concats;

  const real* * p_bottom_data
      = (const real* *)malloc(num_bottoms * sizeof(real*));
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

    top3d->shape[n][0] = top_C;
    top3d->shape[n][1] = H;
    top3d->shape[n][2] = W;
  } // endfor batch

  free(p_bottom_data);

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

  tick1 = clock();
  a_time[6] = (float)(tick1 - tick0) / CLOCKS_PER_SEC;
  a_time[7] += (float)(tick1 - tick0) / CLOCKS_PER_SEC;
}



// --------------------------------------------------------------------------
// layer shape calculator code
// --------------------------------------------------------------------------

void concat_shape(const Tensor* const bottom3d[],
                  Tensor* const top3d,
                  const LayerOption* const option)
{
  const int num_bottoms = option->num_concats;

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
// API code
// --------------------------------------------------------------------------

void forward_concat_layer(void* const net_, void* const layer_)
{
    global_logger.start_log(FORWARD_CONCAT_LAYER);

  Layer* const layer = (Layer*)layer_;

  concat_forward(layer->p_bottoms, &layer->tops[0], &layer->option);

  print_tensor_info(layer->name, &layer->tops[0]);
  #ifdef DEBUG
  {
    for (int i = 0; i < 8; ++i) {
      printf("%4.2f\t", a_time[i] * 1000);
    }
    printf("\n");
  }
  #endif

  global_logger.stop_log(FORWARD_CONCAT_LAYER);
}

void shape_concat_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;

  concat_shape(layer->p_bottoms, &layer->tops[0],
               &layer->option);

  update_net_size(net, layer, 0, 0, 0);
}

void init_concat_layer(void* const net_, void* const layer_,
                       const void* const entry_)
{
  Layer* const layer = (Layer*)layer_;
  LayerOption* const option = &layer->option;

  option->num_concats = layer->num_bottoms;
}



// --------------------------------------------------------------------------
// test code
// --------------------------------------------------------------------------

#ifdef TEST

int main(int argc, char* argv[])
{
  return 0;
}
#endif
