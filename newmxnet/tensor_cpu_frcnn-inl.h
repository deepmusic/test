#ifndef MSHADOW_TENSOR_CPU_FRCNN_INLINE_H
#define MSHADOW_TENSOR_CPU_FRCNN_INLINE_H

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

namespace mshadow {

template<typename Dtype>
inline void ROIPoolForward(Tensor<cpu, 4, Dtype> &out,
                           const Tensor<cpu, 4, Dtype> &data,
                           const Tensor<cpu, 2, Dtype> &bbox,
                           const Tensor<cpu, 4, Dtype> &max_idx,
                           const float spatial_scale) {
  return;
}

template<typename Dtype>
inline void ROIPoolBackward(Tensor<cpu, 4, Dtype> &in_grad,
                            const Tensor<cpu, 4, Dtype> &out_grad,
                            const Tensor<cpu, 2, Dtype> &bbox,
                            const Tensor<cpu, 4, Dtype> &max_idx,
                            const float spatial_scale) {
  return;
}

inline float devIoU(float const * const a, float const * const b) {
  float left = std::max(a[0], b[0]), right = std::min(a[2], b[2]);
  float top = std::max(a[1], b[1]), bottom = std::min(a[3], b[3]);
  float width = std::max(right - left + 1, 0.f), height = std::max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

template<typename Dtype>
void _nms(int* keep_out, int* num_out, const Dtype* boxes_host,
          const int boxes_num, const int boxes_dim, const float nms_overlap_thresh,
          const Tensor<cpu, 1, Dtype> &dummy) {
  int const threadsPerBlock = sizeof(unsigned long long) * 8;

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);
  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  unsigned long long *pmask = &mask_host[0];

  for (int y = 0; y < boxes_num / threadsPerBlock; ++y) {
    for (int x = 0; x < boxes_num / threadsPerBlock; ++y) {
      const int row_size = std::min(boxes_num - y * threadsPerBlock, threadsPerBlock);
      const int col_size = std::min(boxes_num - x * threadsPerBlock, threadsPerBlock);

      Dtype block_boxes[col_size * 5];
      memcpy(block_boxes, &boxes_host[threadsPerBlock * x * 5], col_size * 5 * sizeof(Dtype));

      for (int i = 0; i < row_size; ++i) {
        const int cur_box_idx = threadsPerBlock * y + i;
        const Dtype *cur_box = boxes_host + cur_box_idx * 5;
        unsigned long long t = 0;
        int start = (y == x) ? i + 1: 0;
        for (int j = start; j < col_size; ++j) {
          if (devIoU(cur_box, block_boxes + j * 5) > nms_overlap_thresh) {
            t |= 1ULL << j;
          }
        }
        const int col_blocks = DIVUP(boxes_num, threadsPerBlock);
        pmask[cur_box_idx * col_blocks + x] = t;
      }
    }
  }

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
}

}  // namespace mshadow

#endif  // MSHADOW_TENSOR_CPU_FRCNN_INLINE_H
