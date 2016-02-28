#include "layer.h"

#ifdef GPU
#include "cuda_settings.h"
#endif

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
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < R * C * top_H * top_W;
       index += blockDim.x) {
    // parse thread index -> (r, c, h, w)
    const int r = index / top_W / top_H / C;
    const int c = (index / top_W / top_H) % C;
    const int h = (index / top_W) % top_H;
    const int w = index % top_W;

    // RoI in the bottom plane
    const int x1 = ROUND(roi2d[r * 4 + 0] * spatial_scale);
    const int y1 = ROUND(roi2d[r * 4 + 1] * spatial_scale);
    const int x2 = ROUND(roi2d[r * 4 + 2] * spatial_scale);
    const int y2 = ROUND(roi2d[r * 4 + 3] * spatial_scale);
    const int roi_W = x2 - x1 + 1;
    const int roi_H = y2 - y1 + 1;

    // pooling region for pixel top[r][c][h][w]
    const int hb_start = MIN(H, MAX(0,
                           y1 + (h * roi_H) / top_H));
    const int hb_end = MIN(H, MAX(0,
                           y1 + DIV_THEN_CEIL((h + 1) * roi_H, top_H)));
    const int wb_start = MIN(W, MAX(0,
                           x1 + (w * roi_W) / top_W));
    const int wb_end = MIN(W, MAX(0,
                           x1 + DIV_THEN_CEIL((w + 1) * roi_W, top_W)));

    // if the bottom region is invalid, top[r][c][h][w] = 0
    const bool is_empty = (hb_end <= hb_start) || (wb_end <= wb_start);
    if (is_empty) {
      top4d[index] = 0;
      argmax4d[index] = -1;
    }
    // otherwise, top[r][c][h][w] = "max in the bottom region"
    else {
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
      top4d[index] = maxval;
      argmax4d[index] = maxidx;
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
  // thread index: (r, c, h, w) = r*C*H'*W' + c*H'*W' + h*W' + w
  for (int index = 0; index < R * C * top_H * top_W; ++index) {
    // parse thread index -> (r, c, h, w)
    const int r = index / top_W / top_H / C;
    const int c = (index / top_W / top_H) % C;
    const int h = (index / top_W) % top_H;
    const int w = index % top_W;

    // RoI in the bottom plane
    const int x1 = ROUND(roi2d[r * 4 + 0] * spatial_scale);
    const int y1 = ROUND(roi2d[r * 4 + 1] * spatial_scale);
    const int x2 = ROUND(roi2d[r * 4 + 2] * spatial_scale);
    const int y2 = ROUND(roi2d[r * 4 + 3] * spatial_scale);
    const int roi_W = x2 - x1 + 1;
    const int roi_H = y2 - y1 + 1;

    // pooling region for pixel top[r][c][h][w]
    const int hb_start = MIN(H, MAX(0,
                           y1 + (h * roi_H) / top_H));
    const int hb_end = MIN(H, MAX(0,
                           y1 + DIV_THEN_CEIL((h + 1) * roi_H, top_H)));
    const int wb_start = MIN(W, MAX(0,
                           x1 + (w * roi_W) / top_W));
    const int wb_end = MIN(W, MAX(0,
                           x1 + DIV_THEN_CEIL((w + 1) * roi_W, top_W)));

    // if the bottom region is invalid, top[r][c][h][w] = 0
    const int is_empty = (hb_end <= hb_start) || (wb_end <= wb_start);
    if (is_empty) {
      top4d[index] = 0;
      argmax4d[index] = -1;
    }
    // otherwise, top[r][c][h][w] = "max in the bottom region"
    else {
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
      top4d[index] = maxval;
      argmax4d[index] = maxidx;
    }
  }
}
#endif

// RoI pooling: bottom -> top
//   bottom: C x H x W
//   roi: R x 4
//   top: R x C x H' x W'
//   argmax: R * C * H' * W'
void roipool_forward(const Tensor* const bottom3d,
                     const Tensor* const roi2d,
                     Tensor* const top4d,
                     int* const argmax_data,
                     const ROIPoolOption* option)
{
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
      const int num_blocks = DIV_THEN_CEIL(num_threads, threads_per_block);
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
      const int roi_size = R * 4;
      const int top_size = R * C * top_H * top_W;
      p_bottom_item += bottom_size;
      p_roi_item += roi_size;
      p_top_item += top_size;
      p_argmax_item += top_size;
    }
  } // endfor batch

  top4d->ndim = 4;
  top4d->num_items = bottom3d->num_items;
}

// test code
#ifdef TEST
#include <stdio.h>
#include <stdlib.h>

#define IN_DATA_SIZE 512*36*46
#define OUT_DATA_SIZE 300*512*6*6
#define ROI_SIZE 300*4

int main(int argc, char *argv[])
{
  // variable declaration & memory allocation
  Tensor X, Y, roi;
  real* const X_data = (real*)malloc(IN_DATA_SIZE * sizeof(real));
  real* const roi_data = (real*)malloc(ROI_SIZE*sizeof(real));
  real* const Y_data = (real*)malloc(OUT_DATA_SIZE * sizeof(real));
  real* const Y_true_data = (real*)malloc(OUT_DATA_SIZE * sizeof(real));
  int* p_argmax_data;
  ROIPoolOption option;

  // set option
  {
    option.pooled_height = 6;
    option.pooled_width = 6;
    option.spatial_scale = 0.0625;
  }

  // set data shapes
  {
    X.ndim = 3;
    X.num_items = 1;
    for (int i = 0; i < X.num_items; ++i) {
      X.shape[i][0] = 512;
      X.shape[i][1] = 36;
      X.shape[i][2] = 46;
    }

    roi.ndim = 2;
    roi.num_items = X.num_items;
    for (int i = 0; i < roi.num_items; ++i) {
      roi.shape[i][0] = 300;
      roi.shape[i][1] = 4;
    }

    Y.ndim = X.ndim + 1;
    Y.num_items = X.num_items;
    for (int i = 0; i < Y.num_items; ++i) {
      Y.shape[i][0] = roi.shape[i][0];
      Y.shape[i][1] = X.shape[i][0];
      Y.shape[i][2] = option.pooled_height;
      Y.shape[i][3] = option.pooled_width;
    }
  }

  // load data
  {
    FILE* fp;
    const int X_size = flatten_size(&X);
    const int Y_size = flatten_size(&Y);
    const int roi_size = flatten_size(&roi);

    printf("data loading\n");

    fp = fopen("../data/temp/roipool_bottom0.txt", "r");
    for (int i = 0; i < X_size; ++i) {
      if (fscanf(fp, "%f", &X_data[i]) <= 0) {
        printf("Error occurred while reading roipool_bottom0[%d]\n", i);
      }
    }
    fclose(fp);

    fp = fopen("../data/temp/proposal_top0.txt", "r");
    for (int i = 0; i < roi_size; ++i) {
      if (fscanf(fp, "%f", &roi_data[i]) <= 0) {
        printf("Error occurred while reading proposal_top0[%d]\n", i);
      }
    }
    fclose(fp);

    fp = fopen("../data/temp/roipool_top0.txt", "r");
    for (int i = 0; i < Y_size; ++i) {
      if (fscanf(fp, "%f", &Y_true_data[i]) <= 0) {
        printf("Error occurred while reading roipool_top0[%d]\n", i);
      }
    }
    fclose(fp);
 }

  // CUDA initialization
  #ifdef GPU
  {
    printf("set device\n");
    CUDA_CHECK(cudaSetDevice(0));
  }
  #endif

  // bind loaded data to corresponding tensors
  #ifdef GPU
  {
    const int X_size = flatten_size(&X);
    const int Y_size = flatten_size(&Y);
    const int roi_size = flatten_size(&roi);

    printf("gpu malloc\n");
    CUDA_CHECK(cudaMalloc(&X.data, X_size * sizeof(real)));
    CUDA_CHECK(cudaMalloc(&roi.data, roi_size * sizeof(real)));
    CUDA_CHECK(cudaMalloc(&Y.data, Y_size * sizeof(real)));
    CUDA_CHECK(cudaMalloc(&p_argmax_data, Y_size * sizeof(int)));

    printf("memcpy: cpu -> gpu\n");
    CUDA_CHECK(cudaMemcpy(X.data, X_data, X_size * sizeof(real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(roi.data, roi_data, roi_size * sizeof(real),
                          cudaMemcpyHostToDevice));
  }
  #else
  {
    const int Y_size = flatten_size(&Y);

    X.data = X_data;
    Y.data = Y_data;
    roi.data = roi_data;
    p_argmax_data = (int*)malloc(Y_size * sizeof(int));
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
    const int Y_size = flatten_size(&Y);
    printf("memcpy: cpu <- gpu\n");
    CUDA_CHECK(cudaMemcpy(Y_data, Y.data, Y_size * sizeof(real),
                          cudaMemcpyDeviceToHost));
  }
  #endif

  // verify results
  {
    const int Y_size = flatten_size(&Y);
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
    CUDA_CHECK(cudaFree(X.data));
    CUDA_CHECK(cudaFree(roi.data));
    CUDA_CHECK(cudaFree(Y.data));
    CUDA_CHECK(cudaFree(p_argmax_data));
  }
  #else
  {
    free(p_argmax_data);
  }
  #endif

  return 0;
}
#endif // endifdef TEST
