#ifndef MSHADOW_CUDA_TENSOR_GPU_FRCNN_INLINE_CU
#define MSHADOW_CUDA_TENSOR_GPU_FRCNN_INLINE_CU

#define FRCNN_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define FRCNN_DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
#define FRCNN_NUM_THREADS 1024

namespace mshadow {
namespace cuda {

template<typename Dtype>
__global__ void ROIPoolForwardKernel(const int count, const Dtype* bottom_data,
                                     const float spatial_scale, const int channels, const int height,
                                     const int width, const int pooled_height, const int pooled_width,
                                     const Dtype* bottom_rois, Dtype* top_data, Dtype* argmax_data) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];

    if (roi_batch_ind < 0) {
      top_data[index] = 0;
      argmax_data[index] = 0;
      continue;
    }

    int roi_start_w = round(bottom_rois[1] * spatial_scale);
    int roi_start_h = round(bottom_rois[2] * spatial_scale);
    int roi_end_w = round(bottom_rois[3] * spatial_scale);
    int roi_end_h = round(bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        if (bottom_data[bottom_index] > maxval) {
          maxval = bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = (Dtype)maxidx;
  }
}

template<typename Dtype>
inline void ROIPoolForward(Tensor<gpu, 4, Dtype> &out,
                           const Tensor<gpu, 4, Dtype> &data,
                           const Tensor<gpu, 2, Dtype> &bbox,
                           const Tensor<gpu, 4, Dtype> &max_idx,
                           const float spatial_scale) {
  const Dtype *bottom_data = data.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *top_data = out.dptr_;
  Dtype *argmax_data = max_idx.dptr_;
  const int count = out.shape_.Size();
  const int channels = data.size(1);
  const int height = data.size(2);
  const int width = data.size(3);
  const int pooled_height = out.size(2);
  const int pooled_width = out.size(3);
  dim3 dimGrid((count + FRCNN_NUM_THREADS - 1) / FRCNN_NUM_THREADS);
  dim3 dimBlock(FRCNN_NUM_THREADS);
  //CheckLaunchParam(dimGrid, dimBlock, "ROIPooling");
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  ROIPoolForwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, bottom_data, spatial_scale, channels, height, width,
      pooled_height, pooled_width, bottom_rois, top_data, argmax_data);
  FRCNN_CUDA_CHECK(cudaPeekAtLastError());
}

template<typename Dtype>
__global__ void ROIPoolBackwardKernel(const int count, const Dtype* top_diff,
                                      const Dtype* argmax_data, const int num_rois, const float spatial_scale,
                                      const int channels, const int height, const int width,
                                      const int pooled_height, const int pooled_width, Dtype* bottom_diff,
                                      const Dtype* bottom_rois) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
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

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
      const Dtype* offset_argmax_data = argmax_data + offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = max(roi_end_h - roi_start_h + 1, 1);

      Dtype bin_size_h = static_cast<Dtype>(roi_height)
                         / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = static_cast<Dtype>(roi_width)
                         / static_cast<Dtype>(pooled_width);

      int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
      int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
      int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
      int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);

      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);

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

template<typename Dtype>
inline void ROIPoolBackward(Tensor<gpu, 4, Dtype> &in_grad,
                            const Tensor<gpu, 4, Dtype> &out_grad,
                            const Tensor<gpu, 2, Dtype> &bbox,
                            const Tensor<gpu, 4, Dtype> &max_idx,
                            const float spatial_scale) {
  const Dtype *top_diff = out_grad.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *bottom_diff = in_grad.dptr_;
  Dtype *argmax_data = max_idx.dptr_;
  const int count = in_grad.shape_.Size();
  const int num_rois = bbox.size(0);
  const int channels = in_grad.size(1);
  const int height = in_grad.size(2);
  const int width = in_grad.size(3);
  const int pooled_height = out_grad.size(2);
  const int pooled_width = out_grad.size(3);
  dim3 dimGrid((count + FRCNN_NUM_THREADS - 1) / FRCNN_NUM_THREADS);
  dim3 dimBlock(FRCNN_NUM_THREADS);
  //CheckLaunchParam(dimGrid, dimBlock, "ROIPooling");
  cudaStream_t stream = Stream<gpu>::GetStream(in_grad.stream_);
  ROIPoolBackwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, top_diff, argmax_data, num_rois, spatial_scale, channels, height, width,
      pooled_height, pooled_width, bottom_diff, bottom_rois);
  FRCNN_CUDA_CHECK(cudaPeekAtLastError());
}

static int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

template<typename Dtype>
__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const Dtype *dev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ Dtype block_boxes[threadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const Dtype *cur_box = dev_boxes + cur_box_idx * 5;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = FRCNN_DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

template<typename Dtype>
void _nms_gpu(int* keep_out, int* num_out, const Dtype* boxes_host,
              const int boxes_num, const int boxes_dim, const float nms_overlap_thresh) {
  Dtype* boxes_dev = NULL;
  unsigned long long* mask_dev = NULL;

  const int col_blocks = FRCNN_DIVUP(boxes_num, threadsPerBlock);

  FRCNN_CUDA_CHECK(cudaMalloc(&boxes_dev, boxes_num * boxes_dim * sizeof(Dtype)));
  FRCNN_CUDA_CHECK(cudaMemcpy(boxes_dev,
                              boxes_host,
                              boxes_num * boxes_dim * sizeof(Dtype),
                              cudaMemcpyHostToDevice));

  FRCNN_CUDA_CHECK(cudaMalloc(&mask_dev, boxes_num * col_blocks * sizeof(unsigned long long)));

  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);
  nms_kernel<<<blocks, threads>>>(boxes_num, nms_overlap_thresh, boxes_dev, mask_dev);

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  FRCNN_CUDA_CHECK(cudaMemcpy(&mask_host[0],
                              mask_dev,
                              sizeof(unsigned long long) * boxes_num * col_blocks,
                              cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }
  *num_out = num_to_keep;

  FRCNN_CUDA_CHECK(cudaFree(boxes_dev));
  FRCNN_CUDA_CHECK(cudaFree(mask_dev));
}

} // namespace cuda

template<typename Dtype>
inline void ROIPoolForward(Tensor<gpu, 4, Dtype> &out,
                           const Tensor<gpu, 4, Dtype> &data,
                           const Tensor<gpu, 2, Dtype> &bbox,
                           const Tensor<gpu, 4, Dtype> &max_idx,
                           const float spatial_scale) {
  cuda::ROIPoolForward(out, data, bbox, max_idx, spatial_scale);
}

template<typename Dtype>
inline void ROIPoolBackward(Tensor<gpu, 4, Dtype> &in_grad,
                            const Tensor<gpu, 4, Dtype> &out_grad,
                            const Tensor<gpu, 2, Dtype> &bbox,
                            const Tensor<gpu, 4, Dtype> &max_idx,
                            const float spatial_scale) {
  cuda::ROIPoolBackward(in_grad, out_grad, bbox, max_idx, spatial_scale);
}

template<typename Dtype>
void _nms(int* keep_out, int* num_out, const Dtype* boxes_host,
          const int boxes_num, const int boxes_dim, const float nms_overlap_thresh,
          const Tensor<gpu, 1, Dtype> &dummy) {
  cuda::_nms_gpu(keep_out, num_out, boxes_host, boxes_num, boxes_dim, nms_overlap_thresh);
}

}  // namespace mshadow

#endif  // MSHADOW_CUDA_TENSOR_GPU_FRCNN_INLINE_CU
