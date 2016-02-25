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
    const int hb_start = min(H, max(0,
                           y1 + (h * roi_H) / top_H));
    const int hb_end = min(H, max(0,
                           y1 + DIV_THEN_CEIL((h + 1) * roi_H, top_H)));
    const int wb_start = min(W, max(0,
                           x1 + (w * roi_W) / top_W));
    const int wb_end = min(W, max(0,
                           x1 + DIV_THEN_CEIL((w + 1) * roi_W, top_W)));

    // if the bottom region is invalid, top[r][c][h][w] = 0
    const bool is_empty = (hb_end <= hb_start) || (wb_end <= wb_start);
    if (is_empty) {
      top4d[index] = 0;
      argmax4d[index] = -1;
    }
    // otherwise, top[r][c][h][w] = "max in the bottom region"
    else {
      const real* p_bottom3d = &bottom3d[c * H * W];
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
      const real* p_bottom3d = &bottom3d[c * H * W];
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
  const real* p_bottom_item = bottom3d->data;
  const real* p_roi_item = roi2d->data;
  real* p_top_item = top4d->data;
  int* p_argmax_item = argmax_data;
  for (int n = 0; n < bottom3d->num_items; ++n) {
    const int R = roi2d->shape[n][0];
    const int C = bottom3d->shape[n][0];
    const int H = bottom3d->shape[n][1];
    const int W = bottom3d->shape[n][2];
    const int top_H = option->pooled_height;
    const int top_W = option->pooled_width;
    top4d->shape[n][0] = R;
    top4d->shape[n][1] = C;
    top4d->shape[n][2] = top_H;
    top4d->shape[n][3] = top_W;

    // RoI pooling
    // bottom3d (C x H x W) -> top4d (R x C x H' x W')
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

#ifdef PASS
__global__ void backward_kernel(const int bottom_RCHW, const real* top_diff,
                                      const real* argmax_data, const int num_rois, const float spatial_scale,
                                      const int C, const int H, const int W,
                                      const int top_H, const int top_W, real* bottom_diff,
                                      const real* bottom_rois) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < bottom_RCHW;
       index += blockDim.x * gridDim.x) {
    // (n, c, h, w) coords in bottom data
    int w = index % W;
    int h = (index / W) % H;
    int c = (index / W / H) % C;
    int n = index / W / H / C;

    real gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const real* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
      int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
      int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
      int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                           h >= roi_start_h && h <= roi_end_h);
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * C + c) * top_H * top_W;
      const real* offset_top_diff = top_diff + offset;
      const real* offset_argmax_data = argmax_data + offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = max(roi_end_h - roi_start_h + 1, 1);

      real bin_size_h = static_cast<real>(roi_height)
                         / static_cast<real>(top_H);
      real bin_size_w = static_cast<real>(roi_width)
                         / static_cast<real>(top_W);

      int phstart = floor(static_cast<real>(h - roi_start_h) / bin_size_h);
      int phend = ceil(static_cast<real>(h - roi_start_h + 1) / bin_size_h);
      int pwstart = floor(static_cast<real>(w - roi_start_w) / bin_size_w);
      int pwend = ceil(static_cast<real>(w - roi_start_w + 1) / bin_size_w);

      phstart = min(max(phstart, 0), top_H);
      phend = min(max(phend, 0), top_H);
      pwstart = min(max(pwstart, 0), top_W);
      pwend = min(max(pwend, 0), top_W);

      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (round(offset_argmax_data[ph * pooled_width + pw]) == (h * width + w)) {
            gradient += offset_top_diff[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

void backward(Tensor<gpu, 4, real> &in_grad,
                            const Tensor<gpu, 4, real> &out_grad,
                            const Tensor<gpu, 2, real> &bbox,
                            const Tensor<gpu, 4, real> &max_idx,
                            const float spatial_scale) {
  const real *top_diff = out_grad.dptr_;
  const real *bottom_rois = bbox.dptr_;
  real *bottom_diff = in_grad.dptr_;
  real *argmax_data = max_idx.dptr_;
  const int count = in_grad.shape_.Size();
  const int num_rois = bbox.size(0);
  const int C = in_grad.size(1);
  const int H = in_grad.size(2);
  const int W = in_grad.size(3);
  const int top_H = out_grad.size(2);
  const int top_W = out_grad.size(3);
  int dimGrid = (count + FRCNN_NUM_THREADS - 1) / FRCNN_NUM_THREADS;
  int dimBlock = FRCNN_NUM_THREADS;
  backward_kernel<<<dimGrid, dimBlock>>>(
      count, top_diff, argmax_data, num_rois, spatial_scale, C, H, W,
      top_H, top_W, bottom_diff, bottom_rois);
  FRCNN_CUDA_CHECK(cudaPeekAtLastError());
}
#endif

#include <stdio.h>
#include <stdlib.h>

#define IN_DATA_SIZE 512*36*46
#define OUT_DATA_SIZE 300*512*6*6
#define ROI_SIZE 300*4

int main(void)
{
  Tensor X, Y, roi;

  real* X_data = (real*)malloc(IN_DATA_SIZE * sizeof(real));
  real* roi_data = (real*)malloc(ROI_SIZE*sizeof(real));
  real* Y_data = (real*)malloc(OUT_DATA_SIZE * sizeof(real));
  real* Y_true_data = (real*)malloc(OUT_DATA_SIZE * sizeof(real));
  int* p_argmax_data;
  ROIPoolOption option;

  {
    option.pooled_height = 6;
    option.pooled_width = 6;
    option.spatial_scale = 0.0625;
  }

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

  {
    int X_size = flatten_size(&X);
    int Y_size = flatten_size(&Y);
    int roi_size = flatten_size(&roi);

    FILE* fp = fopen("../data/temp/roipool_bottom0.txt", "r");
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

#ifdef GPU
  {
    printf("set device\n");
    CUDA_CHECK(cudaSetDevice(0));
  }
#endif

#ifdef GPU
  {
    int X_size = flatten_size(&X);
    int Y_size = flatten_size(&Y);
    int roi_size = flatten_size(&roi);

    printf("cuda malloc");
    CUDA_CHECK(cudaMalloc(&X.data, X_size*sizeof(real)));
    CUDA_CHECK(cudaMalloc(&roi.data, roi_size*sizeof(real)));
    CUDA_CHECK(cudaMalloc(&Y.data, Y_size*sizeof(real)));
    CUDA_CHECK(cudaMalloc(&p_argmax_data, Y_size*sizeof(int)));

    printf("memcopy\n");
    CUDA_CHECK(cudaMemcpy(X.data, X_data, X_size*sizeof(real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(roi.data, roi_data, roi_size*sizeof(real), cudaMemcpyHostToDevice));
  }
#else
  {
    int argmax_size = flatten_size(&Y);

    X.data = &X_data[0];
    Y.data = &Y_data[0];
    roi.data = &roi_data[0];
    p_argmax_data = (int*)malloc(argmax_size * sizeof(int));
  }
#endif

  {
    printf("do forward\n");
    roipool_forward(&X, &roi, &Y, p_argmax_data, &option);
  }

#ifdef GPU
  {
    int Y_size = flatten_size(&Y);
    printf("memcpy");
    CUDA_CHECK(cudaMemcpy(Y_data, Y.data, Y_size*sizeof(real), cudaMemcpyDeviceToHost));
  }
#endif

  {
    int Y_size = flatten_size(&Y);

    for (int i = 0; i < Y_size; ++i) {
      if (Y_data[i] != Y_true_data[i]) {
        printf("Y[%d] = %.6f  Y_true[%d] = %.6f\n", i, Y_data[i], i, Y_true_data[i]);
      }
    }
  }

  {
    printf("free\n");
    free(X_data);
    free(roi_data);
    free(Y_data);
    free(Y_true_data);
  }
#ifdef GPU
  {
    printf("cuda free\n");
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
