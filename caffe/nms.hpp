#ifndef _CAFFE_UTIL_NMS_HPP_
#define _CAFFE_UTIL_NMS_HPP_

#include <vector>

namespace caffe {

template <typename Dtype>
void nms_cpu(const int num_boxes,
             const Dtype boxes_cpu[],
             int index_out[],
             int* const num_out,
             const int base_index,
             const Dtype nms_thresh,
             const int max_num_out);

template <typename Dtype>
void nms_gpu(const int num_boxes,
             const Dtype boxes_gpu[],
             int index_out[],
             int* const num_out,
             const int base_index,
             const Dtype nms_thresh,
             const int max_num_out);

}  // namespace caffe

#endif  // CAFFE_UTIL_NMS_HPP_
