#include "layer.h"
#include "cuda_settings.h"

typedef struct ROIPoolOption_
{
  int pooled_height;
  int pooled_width;
  real spatial_scale;
} ROIPoolOption;

// RoI pooling bottom3d (C x H x W) -> top4d (R x C x H' x W')
//   given pixel (r, c, h, w) at top4d and RoI (x1, y1,, x2, y2),
//     top4d[r][c][h][w] = max_{hb,wb}{ bottom3d[c][hb][wb] }
//       hb, wb: pooling region corresponding to (h, w)
void roi_pool(const real* bottom3d, const real* roi2d,
                         real* const top4d, int* const argmax4d,
                         const int R, const int C, const int H, const int W,
                         const int top_H, const int top_W,
                         const real spatial_scale)
{
  const int top_RCHW = R * C * top_H * top_W;

  // thread index: (r, c, h, w) = r*C*H'*W' + c*H'*W' + h*W' + w
  //for (int index = blockIdx.x * blockDim.x + threadIdx.x;
  //   index < top_RCHW;
  //     index += blockDim.x) {
  for (int index = 0; index < top_RCHW; ++index) {
    // parse thread index -> (r, c, h, w)
    const int r = index / top_W / top_H / C;
    const int c = (index / top_W / top_H) % C;
    const int h = (index / top_W) % top_H;
    const int w = index % top_W;
#ifdef PASS
    // RoI in the bottom plane
    const real x1 = roi2d[r * 4 + 0] * spatial_scale;
    const real y1 = roi2d[r * 4 + 1] * spatial_scale;
    const real roi_W = roi2d[r * 4 + 2] * spatial_scale - x1; // x2 - x1
    const real roi_H = roi2d[r * 4 + 3] * spatial_scale - y1; // y2 - y1

    // pooling region for pixel top[r][c][h][w]
    const int hb_start = min(H - 1, max(0, (int)floor(y1 + h * roi_H / top_H)));
    const int hb_end = min(H, max(1, (int)ceil(y1 + (h + 1.0f) * roi_H / top_H)));
    const int wb_start = min(W - 1, max(0, (int)floor(x1 + w * roi_W / top_W)));
    const int wb_end = min(W, max(1, (int)ceil(x1 + (w + 1.0f) * roi_W / top_W)));
#else
    int roi_start_w = round(roi2d[r * 4 + 0] * spatial_scale);
    int roi_start_h = round(roi2d[r * 4 + 1] * spatial_scale);
    int roi_end_w = round(roi2d[r * 4 + 2] * spatial_scale);
    int roi_end_h = round(roi2d[r * 4 + 3] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    real bin_size_h = (real)(roi_height)
                       / (real)(top_H);
    real bin_size_w = (real)(roi_width)
                       / (real)(top_W);

    int hstart = static_cast<int>(floor((real)(h)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor((real)(w)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil((real)(h + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil((real)(w + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    int hb_start = min(max(hstart + roi_start_h, 0), H);
    int hb_end = min(max(hend + roi_start_h, 0), H);
    int wb_start = min(max(wstart + roi_start_w, 0), W);
    int wb_end = min(max(wend + roi_start_w, 0), W);
    bool is_empty = (hb_end <= hb_start) || (wb_end <= wb_start);
#endif
    // max pooling
    real maxval = 0;
    int maxidx = -1;
    const real* p_bottom_data = &bottom3d[c * H * W];
    for (int hb = hb_start; hb < hb_end; ++hb) {
      for (int wb = wb_start; wb < wb_end; ++wb) {
        const int bottom_index = hb * W + wb;
        if (p_bottom_data[bottom_index] > maxval) {
          maxval = p_bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      }
    }
    top4d[index] = maxval;
    argmax4d[index] = maxidx;
  }
}

void forward(const Tensor* bottom3d, const Tensor* roi2d,
             Tensor* const top4d, int* const argmax_data,
             const ROIPoolOption* option)
{
  const real* p_bottom_data = bottom3d->data;
  const real* p_roi_data = roi2d->data;
  real* p_top_data = top4d->data;
  int* p_argmax_data = argmax_data;
  for (int n = 0; n < bottom3d->num_items; ++n) {
    const int R = roi2d->shape[n][0];
    const int C = bottom3d->shape[n][0];
    const int H = bottom3d->shape[n][1];
    const int W = bottom3d->shape[n][2];
    const int top_H = option->pooled_height;
    const int top_W = option->pooled_width;
    const int top_RCHW = R * C * top_H * top_W;
    top4d->shape[n][0] = R;
    top4d->shape[n][1] = C;
    top4d->shape[n][2] = top_H;
    top4d->shape[n][3] = top_W;

   { // RoI pooling
    // bottom3d (C x H x W) -> top4d (R x C x H' x W')
    const int num_threads = 1024;
    const int num_blocks = (num_threads - 1 + top_RCHW) / num_threads;
    //roi_pool<<<num_blocks, num_threads>>>(
    roi_pool(
        p_bottom_data, p_roi_data, p_top_data, p_argmax_data,
        R, C, H, W, top_H, top_W, option->spatial_scale);
    //CUDA_CHECK(cudaPeekAtLastError());
   } // end RoI pooling

    // locate next data
    p_bottom_data += C * H * W;
    p_roi_data += R * 4;
    p_top_data += top_RCHW;
    p_argmax_data += top_RCHW;
  } // endfor batch
  top4d->num_items = bottom3d->num_items;
  top4d->ndim = 4;
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

#include <stdlib.h>

int main(void)
{
  ROIPoolOption option;
  option.pooled_height = 6;
  option.pooled_width = 6;
  option.spatial_scale = 0.0625;

  Tensor convf, roi, out;

  real* convf_data = (real*)malloc(512*36*46*sizeof(real));
  real* roi_data = (real*)malloc(300*4*sizeof(real));
  real* out_data = (real*)malloc(300*512*6*6*sizeof(real));
  int* argmax_data = (int*)malloc(300*512*6*6*sizeof(int));

  convf.ndim = 3; convf.num_items = 1; convf.data = convf_data;
  for (int i = 0; i < convf.num_items; ++i) {
    convf.shape[i][0] = 512;
    convf.shape[i][1] = 36;
    convf.shape[i][2] = 46;
  }
  roi.ndim = 2; roi.num_items = convf.num_items; roi.data = roi_data;
  for (int i = 0; i < roi.num_items; ++i) {
    roi.shape[i][0] = 300;
    roi.shape[i][1] = 4;
  }
  out.ndim = 4; out.num_items = convf.num_items; out.data = out_data;

  FILE* fp = fopen("convf.txt", "r");
  for (int i = 0; i < flatten_size(&convf); ++i)
    fscanf(fp, "%f", &convf_data[i]);
  fclose(fp);
  fp = fopen("roi.txt", "r");
  for (int i = 0; i < flatten_size(&roi); ++i)
    fscanf(fp, "%f", &roi_data[i]);
  fclose(fp);

  forward(&convf, &roi, &out, &argmax_data[0], &option);

  real* p_out_data = out.data;
  for (int n = 0; n < out.num_items; ++n) {
    printf("batch %d: %d x %d x %d x %d\n", n, out.shape[n][0], out.shape[n][1], out.shape[n][2], out.shape[n][3]);
    for (int m = 0; m < out.shape[n][0]; ++m) {
      for (int c = 0; c < out.shape[n][1]; ++c) {
        for (int h = 0; h < out.shape[n][2]; ++h) {
          for (int w = 0; w < out.shape[n][3]; ++w) {
            printf("%.6f\n", *(p_out_data++));
          }
        }
      }
    }
  }

  free(convf_data);
  free(roi_data);
  free(out_data);
  free(argmax_data);
  return 0;
}
