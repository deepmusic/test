#include "layer.h"

#include <time.h>

static float a_time[8] = { 0, };
static clock_t tick0, tick1;

// --------------------------------------------------------------------------
// kernel code
//   roi_pool_{gpu, cpu}
// --------------------------------------------------------------------------

// RoI pooling bottom3d (C x H x W) -> top4d (R x C x H' x W')
//   given pixel (r, c, h, w) at top4d and RoI (x1, y1,, x2, y2),
//     top4d[r][c][h][w] = max_{hb,wb}{ bottom3d[c][hb][wb] }
//       hb, wb: pooling region corresponding to (h, w)
#ifdef GPU
__global__
void roi_pool_gpu(const real* const bottom3d,
                  const real* const roi2d,
                  real* const top4d,
                  int* const argmax4d,
                  const int R, const int C, const int H, const int W,
                  const int top_H, const int top_W,
                  const real spatial_scale)
{
  // thread index: (r, c, h, w) = r*C*H'*W' + c*H'*W' + h*W' + w
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < R * C * top_W * top_H) {
    // parse thread index -> (r, c, h, w)
    const int r = index / top_W / top_H / C;
    const int c = (index / top_W / top_H) % C;
    const int h = (index / top_W) % top_H;
    const int w = index % top_W;

    // RoI in the bottom plane
    const int x1 = ROUND(roi2d[r * 5 + 0] * spatial_scale);
    const int y1 = ROUND(roi2d[r * 5 + 1] * spatial_scale);
    const int x2 = ROUND(roi2d[r * 5 + 2] * spatial_scale);
    const int y2 = ROUND(roi2d[r * 5 + 3] * spatial_scale);
    const int roi_W = x2 - x1 + 1;
    const int roi_H = y2 - y1 + 1;

    // pooling region for pixel top[r][c][h][w]
    const int hb_start = MIN(H,  MAX(0,
                           y1 + (h * roi_H) / top_H));
    const int hb_end = MIN(H,  MAX(0,
                           y1 + DIV_THEN_CEIL((h + 1) * roi_H,  top_H)));
    const int wb_start = MIN(W,  MAX(0,
                           x1 + (w * roi_W) / top_W));
    const int wb_end = MIN(W,  MAX(0,
                           x1 + DIV_THEN_CEIL((w + 1) * roi_W,  top_W)));

    // find maximum in the bottom region
    const real* p_bottom3d = bottom3d + c * H * W;
    int maxidx = hb_start * W + wb_start;
    real maxval = p_bottom3d[maxidx];
    for (int hb = hb_start; hb < hb_end; ++hb) {
      for (int wb = wb_start; wb < wb_end; ++wb) {
        const int bottom_index = hb * W + wb;
        if (p_bottom3d[bottom_index] > maxval) {
          maxval = p_bottom3d[bottom_index];
          maxidx = bottom_index;
        }
      }
    }

    // if the bottom region is not empty,
    //   top[r][c][h][w] = "max in the region"
    // otherwise, assign 0
    {
      const int not_empty = (hb_start < hb_end) * (wb_start < wb_end);
      top4d[index] = not_empty * maxval;
      #ifdef BACKWARD
      argmax4d[index] = not_empty * maxidx + (1 - not_empty) * (-1);
      #endif
    }
  }
}
#else
void roi_pool_cpu(const real* const bottom3d,
                  const real* const roi2d,
                  real* const top4d,
                  int* const argmax4d,
                  const int R, const int C, const int H, const int W,
                  const int top_H, const int top_W,
                  const real spatial_scale)
{
  const int top_area = top_H * top_W;
  const int top_volume = C * top_H * top_W;

  for (int r = 0; r < R; ++r) {
    // RoI in the bottom plane
    const int x1 = ROUND(roi2d[r * 5 + 0] * spatial_scale);
    const int y1 = ROUND(roi2d[r * 5 + 1] * spatial_scale);
    const int x2 = ROUND(roi2d[r * 5 + 2] * spatial_scale);
    const int y2 = ROUND(roi2d[r * 5 + 3] * spatial_scale);
    const int roi_W = x2 - x1 + 1;
    const int roi_H = y2 - y1 + 1;

    for (int h = 0; h < top_H; ++h) {
    for (int w = 0; w < top_W; ++w) {
      // pooling region for pixel top[r][c][h][w]
      const int hb_start = MIN(H,  MAX(0,
                               y1 + (h * roi_H) / top_H));
      const int hb_end = MIN(H,  MAX(0,
                             y1 + DIV_THEN_CEIL((h + 1) * roi_H,  top_H)));
      const int wb_start = MIN(W,  MAX(0,
                               x1 + (w * roi_W) / top_W));
      const int wb_end = MIN(W,  MAX(0,
                             x1 + DIV_THEN_CEIL((w + 1) * roi_W,  top_W)));

      const int top_index = r * top_volume + h * top_W + w;

      // if the bottom region is empty,
      //   top[r][c][h][w] = 0
      if (hb_start >= hb_end || wb_start >= wb_end) {
        for (int c = 0; c < C; ++c) {
          top4d[top_index + c * top_area] = 0;
          #ifdef BACKWARD
          argmax4d[top_index + c * top_area] = -1;
          #endif
        }
        continue;
      }

      // if the bottom region is not empty,
      //   top[r][c][h][w] = "max in the region"
      for (int c = 0; c < C; ++c) {
        // find maximum in the bottom region
        const real* p_bottom3d = bottom3d + c * H * W;
        int maxidx = hb_start * W + wb_start;
        for (int hb = hb_start; hb < hb_end; ++hb) {
          for (int wb = wb_start; wb < wb_end; ++wb) {
            maxidx = (p_bottom3d[hb * W + wb] > p_bottom3d[maxidx]) ?
                      hb * W + wb : maxidx;
          }
        }
        top4d[top_index + c * top_area] = p_bottom3d[maxidx];
        #ifdef BACKWARD
        argmax4d[top_index + c * top_area] = maxidx;
        #endif
      } // endfor c
    }} // endfor h, w
  } // endfor r
}
#endif



// --------------------------------------------------------------------------
// layer operator code
//   roipool_forward
// --------------------------------------------------------------------------

// RoI pooling: bottom -> top
//   bottom: C x H x W
//   roi: R x 5
//   top: R x C x H' x W'
//   argmax: R * C * H' * W' array
void roipool_forward(const Tensor* const bottom3d,
                     const Tensor* const roi2d,
                     Tensor* const top4d,
                     int* const argmax_data,
                     const LayerOption* option)
{
  tick0 = clock();

  // top height & width
  const int top_H = option->pooled_height; // H'
  const int top_W = option->pooled_width; // W'

  // do forward-pass for each item in the batch
  const real* p_bottom_item = bottom3d->data;
  const real* p_roi_item = roi2d->data;
  real* p_top_item = top4d->data;
  int* p_argmax_item = argmax_data;
  for (int n = 0; n < bottom3d->num_items; ++n) {
    // bottom shape: R x C x H X W
    const int R = roi2d->shape[n][0];
    const int C = bottom3d->shape[n][0];
    const int H = bottom3d->shape[n][1];
    const int W = bottom3d->shape[n][2];

    // set top shape: R x C x H' x W'
    top4d->shape[n][0] = R;
    top4d->shape[n][1] = C;
    top4d->shape[n][2] = top_H;
    top4d->shape[n][3] = top_W;

    // RoI pooling
    //   bottom3d (C x H x W) -> top4d (R x C x H' x W')
    {
    #ifdef GPU
      const int num_threads = R * C * top_H * top_W;
      const int threads_per_block = 512;
      const int num_blocks = DIV_THEN_CEIL(num_threads,  threads_per_block);
      roi_pool_gpu<<<num_blocks, threads_per_block>>>(
          p_bottom_item,  p_roi_item,  p_top_item,  p_argmax_item,
          R,  C,  H,  W,  top_H,  top_W,  option->spatial_scale);
    #else
      roi_pool_cpu(
          p_bottom_item,  p_roi_item,  p_top_item,  p_argmax_item,
          R,  C,  H,  W,  top_H,  top_W,  option->spatial_scale);
    #endif
    }

    // locate next item
    {
      const int bottom_size = C * H * W;
      const int roi_size = R * 5;
      const int top_size = R * C * top_H * top_W;
      p_bottom_item += bottom_size;
      p_roi_item += roi_size;
      p_top_item += top_size;
      p_argmax_item += top_size;
    }
  } // endfor batch

  // if option->flatten = true,
  // reshape to 2d tensor: total_num_rois x (C * H' * W')
  if (option->flatten) {
    // for all items, C should be equal to each other
    const int C = bottom3d->shape[0][0];

    // calculate total number of RoI-pooled data
    int total_num_rois = 0;
    for (int n = 0; n < roi2d->num_items; ++n) {
      total_num_rois += roi2d->shape[n][0];
    }

    // reshape to 2d tensor: total_num_rois x (C * H' * W')
    top4d->ndim = 2;
    top4d->num_items = 1;
    top4d->shape[0][0] = total_num_rois;
    top4d->shape[0][1] = C * top_H * top_W;
    top4d->start[0] = 0;
  }
  else {
    top4d->ndim = 4;
    top4d->num_items = bottom3d->num_items;
    {
      int total_size = 0;
      for (int n = 0; n < bottom3d->num_items; ++n) {
        const int R = roi2d->shape[n][0];
        const int C = bottom3d->shape[n][0];
        const int top_size = R * C * top_H * top_W;
        top4d->start[n] = total_size;
        total_size += top_size;
      }
    }
  }

  tick1 = clock();
  a_time[6] = (float)(tick1 - tick0) / CLOCKS_PER_SEC;
  a_time[7] += (float)(tick1 - tick0) / CLOCKS_PER_SEC;
}



// --------------------------------------------------------------------------
// layer shape calculator code
// --------------------------------------------------------------------------

void roipool_shape(const Tensor* const bottom3d,
                   const Tensor* const roi2d,
                   Tensor* const top4d,
                   int* const argmax_size,
                   const LayerOption* option)
{
  // top height & width
  const int top_H = option->pooled_height; // H'
  const int top_W = option->pooled_width; // W'

  // if option->flatten = true,
  // reshape to 2d tensor: total_num_rois x (C * H' * W')
  if (option->flatten) {
    // for all items, C should be equal to each other
    const int C = bottom3d->shape[0][0];

    // calculate total number of RoI-pooled data
    int total_num_rois = 0;
    for (int n = 0; n < roi2d->num_items; ++n) {
      total_num_rois += roi2d->shape[n][0];
    }

    // reshape to 2d tensor: total_num_rois x (C * H' * W')
    top4d->ndim = 2;
    top4d->num_items = 1;
    top4d->shape[0][0] = total_num_rois;
    top4d->shape[0][1] = C * top_H * top_W;
    top4d->start[0] = 0;

    // argmax data size = total top size
    *argmax_size = top4d->shape[0][0] * top4d->shape[0][1];

    return;
  }

  // otherwise, calculate shape for each item in the batch
  for (int n = 0; n < bottom3d->num_items; ++n) {
    // bottom shape: R x C x H X W
    const int R = roi2d->shape[n][0];
    const int C = bottom3d->shape[n][0];

    // top shape: R x C x H' x W'
    top4d->shape[n][0] = R;
    top4d->shape[n][1] = C;
    top4d->shape[n][2] = top_H;
    top4d->shape[n][3] = top_W;
  }

  top4d->ndim = 4;
  top4d->num_items = bottom3d->num_items;
  {
    int total_size = 0;
    for (int n = 0; n < bottom3d->num_items; ++n) {
      const int R = roi2d->shape[n][0];
      const int C = bottom3d->shape[n][0];
      const int top_size = R * C * top_H * top_W;
      top4d->start[n] = total_size;
      total_size += top_size;
    }

    // argmax data size = total top size
    *argmax_size = total_size;
  }
}



// --------------------------------------------------------------------------
// API code
// --------------------------------------------------------------------------

void forward_roipool_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;

  roipool_forward(layer->p_bottoms[0], layer->p_bottoms[1],
                  &layer->tops[0],
                  net->tempint_data, &layer->option);
  print_tensor_info(layer->name, &layer->tops[0]);
  #ifdef DEBUG
  {
    for (int i = 0; i < 8; ++i) {
      printf("%4.2f\t", a_time[i] * 1000);
    }
    printf("\n");
  }
  #endif
}

void shape_roipool_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;

  int tempint_size;

  roipool_shape(layer->p_bottoms[0], layer->p_bottoms[1], &layer->tops[0],
                &tempint_size, &layer->option);

  update_net_size(net, layer, 0, tempint_size, 0);
}



// --------------------------------------------------------------------------
// test code
// --------------------------------------------------------------------------

#ifdef TEST

int main(int argc, char* argv[])
{
  // variable declaration & memory allocation
  Tensor X, Y, roi;
  real *X_data = NULL, *Y_data = NULL, *Y_true_data = NULL;
  real *roi_data = NULL;
  int* p_argmax_data = NULL;
  LayerOption option;
  int argmax_size;

  // set option
  {
    option.pooled_height = 6;
    option.pooled_width = 6;
    option.spatial_scale = 0.0625;
    option.flatten = 0;
  }

  // load data
  {
    int ndim;
    int shape[g_max_ndim];
    int total_size;

    X_data = load_data("../data/temp/roipool_bottom0.bin",
                       &ndim, shape, NULL);
    X.num_items = shape[0];
    X.ndim = ndim - 1;
    total_size = 0;
    for (int n = 0; n < X.num_items; ++n) {
      int size_n = 1;
      for (int i = 0; i < X.ndim; ++i) {
        X.shape[n][i] = shape[i + 1];
        size_n *= shape[i + 1];
      }
      X.start[n] = total_size;
      total_size += size_n;
    }

    roi_data = load_data("../data/temp/roipool_bottom1.bin",
                         &ndim, shape, NULL);
    roi.num_items = X.num_items;
    roi.ndim = 2;
    for (int n = 0; n < roi.num_items; ++n) {
      roi.shape[n][1] = 5;
    }
    {
      const int num_rois = shape[0];
      for (int i = 0; i < num_rois; ++i) {
        const int n = (int)ROUND(roi_data[i * 5 + 0]);
        const real x1 = roi_data[i * 5 + 1];
        const real y1 = roi_data[i * 5 + 2];
        const real x2 = roi_data[i * 5 + 3];
        const real y2 = roi_data[i * 5 + 4];
        ++roi.shape[n][0];
        roi_data[i * 5 + 0] = x1;
        roi_data[i * 5 + 1] = y1;
        roi_data[i * 5 + 2] = x2;
        roi_data[i * 5 + 3] = y2;
      }
    }

    roipool_shape(&X, &roi, &Y, &argmax_size, &option);

    Y_true_data = load_data("../data/temp/roipool_top0.bin",
                            &ndim, shape, NULL);
    Y_data = (real*)malloc(flatten_size(&Y) * sizeof(real));
  }

  // CUDA initialization
  #ifdef GPU
  {
    printf("set device\n");
    cudaSetDevice(0);
  }
  #endif

  // bind loaded data to corresponding tensors
  #ifdef GPU
  {
    const long int X_size = flatten_size(&X);
    const long int Y_size = flatten_size(&Y);
    const long int roi_size = flatten_size(&roi);

    printf("gpu malloc\n");
    cudaMalloc(&X.data, X_size * sizeof(real));
    cudaMalloc(&roi.data, roi_size * sizeof(real));
    cudaMalloc(&Y.data, Y_size * sizeof(real));
    cudaMalloc(&p_argmax_data, argmax_size * sizeof(int));

    printf("memcpy: cpu -> gpu\n");
    cudaMemcpyAsync(X.data, X_data, X_size * sizeof(real),
                    cudaMemcpyHostToDevice);
    cudaMemcpyAsync(roi.data, roi_data, roi_size * sizeof(real),
                    cudaMemcpyHostToDevice);
  }
  #else
  {
    X.data = X_data;
    Y.data = Y_data;
    roi.data = roi_data;
    p_argmax_data = (int*)malloc(argmax_size * sizeof(int));
  }
  #endif

  // do forward operation
  {
    printf("do forward\n");
    roipool_forward(&X, &roi, &Y, p_argmax_data, &option);
  }

  // copy GPU data to main memory
  #ifdef GPU
  {
    const long int Y_size = flatten_size(&Y);

    printf("memcpy: cpu <- gpu\n");
    cudaMemcpyAsync(Y_data, Y.data, Y_size * sizeof(real),
                    cudaMemcpyDeviceToHost);
  }
  #endif

  // verify results
  {
    const long int Y_size = flatten_size(&Y);

    printf("verification\n");

    for (int i = 0; i < Y_size; ++i) {
      if (Y_data[i] != Y_true_data[i]) {
        printf("Y[%d] = %.6f  Y_true[%d] = %.6f\n",
               i, Y_data[i], i, Y_true_data[i]);
      }
    }
  }

  // memory deallocation
  {
    printf("free\n");
    free(X_data);
    free(roi_data);
    free(Y_data);
    free(Y_true_data);
  }
  #ifdef GPU
  {
    printf("gpu free\n");
    cudaFree(X.data);
    cudaFree(roi.data);
    cudaFree(Y.data);
    cudaFree(p_argmax_data);
  }
  #else
  {
    free(p_argmax_data);
  }
  #endif

  return 0;
}
#endif // endifdef TEST
