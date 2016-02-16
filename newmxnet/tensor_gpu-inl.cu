/*!
 *  Copyright (c) 2014 by Contributors
 * \file tensor_gpu-inl.cuh
 * \brief implementation of GPU code using CUDA
 * \author Bing Xu, Tianqi Chen
 */
#ifndef MSHADOW_CUDA_TENSOR_GPU_INL_CUH_
#define MSHADOW_CUDA_TENSOR_GPU_INL_CUH_
#include "../tensor.h"
#include "./reduce.cuh"

namespace mshadow {
namespace cuda {
/* load unit for memory access, if CUDAARCH not defined, this is advanced nvcc */
#if MSHADOW_OLD_CUDA
const int kMemUnitBits = 4;
const int kMaxThreadsPerBlock = 512;
#else
const int kMemUnitBits = 5;
const int kMaxThreadsPerBlock = 1024;
#endif
/*! \brief number of units that can do synchronized update, half warp size */
const int kMemUnit = 1 << kMemUnitBits;
/*! \brief mask that could be helpful sometime */
const int kMemUnitMask = kMemUnit - 1;
/*! \brief suggested thread number(logscale) for mapping kernel */
const int kBaseThreadBits = 8;
/*! \brief suggested thread number for mapping kernel */
const int kBaseThreadNum  = 1 << kBaseThreadBits;
/*! \brief maximum value of grid */
const int kMaxGridNum = 65535;
/*! \brief suggested grid number for mapping kernel */
const int kBaseGridNum = 1024;
/*! \brief get align stride for given size in x dimension */
inline index_t GetAlignStride(index_t xsize) {
  if (xsize >= MSHADOW_MIN_PAD_RATIO * 32) {
    return ((xsize  + kMemUnit - 1) >> kMemUnitBits) << kMemUnitBits;
  } else {
    // if originally space is not aligned, no necessary to to alligned thread allocation
    return xsize;
  }
}
inline void CheckLaunchParam(dim3 dimGrid, dim3 dimBlock, const char *estr = "") {
  if (dimBlock.x * dimBlock.y * dimBlock.z > static_cast<unsigned>(kMaxThreadsPerBlock) ||
      dimGrid.x > 65535 || dimGrid.y > 65535) {
    LOG(FATAL) << "too large launch parameter: "
      << estr << "["
      << dimBlock.x << ","
      << dimBlock.y << ","
      << dimBlock.z << "]";
  }
}
template<typename Saver, typename DstPlan,
         typename Plan, int block_dim_bits>
__device__ void MapPlanProc(DstPlan dst, index_t xstride,
                            Shape<2> dshape, const Plan exp, int block_idx) {
  const index_t tid = (block_idx << block_dim_bits) + threadIdx.x;
  const int y = tid / xstride;
  const int x = tid % xstride;
  if (y < dshape[0] && x < dshape[1]) {
    Saver::Save(dst.REval(y, x), exp.Eval(y,x));
  }
}
template<typename Saver,int block_dim_bits,
         typename DstPlan, typename Plan>
__global__ void MapPlanKernel(DstPlan dst, index_t xstride,
                              Shape<2> dshape, const Plan exp) {
  MapPlanProc<Saver, DstPlan, Plan, block_dim_bits>
      (dst, xstride, dshape, exp, blockIdx.x);
}
template<typename Saver, int block_dim_bits, int grid_size,
         typename DstPlan, typename Plan>
__global__ void MapPlanLargeKernel(DstPlan dst, index_t xstride,
                                   Shape<2> dshape, const Plan exp, int repeat) {
  for (int i = 0; i < repeat; ++i) {
  MapPlanProc<Saver, DstPlan, Plan, block_dim_bits>
      (dst, xstride, dshape, exp, blockIdx.x + i * grid_size);
  }
}

template<typename Saver, typename DstExp, typename E, typename DType>
inline void MapPlan(expr::Plan<DstExp, DType> dst,
                    const expr::Plan<E, DType> &plan,
                    Shape<2> dshape,
                    cudaStream_t stream) {
  const index_t xstride = GetAlignStride(dshape[1]);
  const int num_block = (dshape[0] * xstride + kBaseThreadNum-1) / kBaseThreadNum;
  dim3 dimBlock(kBaseThreadNum, 1, 1);

  if (num_block < kMaxGridNum) {
    dim3 dimGrid(num_block, 1, 1);
    MapPlanKernel<Saver, kBaseThreadBits,
                  expr::Plan<DstExp, DType>,
                  expr::Plan<E, DType> >
        <<<dimGrid, dimBlock, 0, stream>>>(dst, xstride, dshape, plan);
  } else {
    int repeat = (num_block + kBaseGridNum-1) / kBaseGridNum;
    dim3 dimGrid(kBaseGridNum, 1 , 1);
    MapPlanLargeKernel<Saver, kBaseThreadBits, kBaseGridNum,
                       expr::Plan<DstExp, DType>,
                       expr::Plan<E, DType> >
        <<<dimGrid, dimBlock, 0, stream>>>(dst, xstride, dshape, plan, repeat);
  }
}

template<typename Saver,typename Reducer, int warp_bits,
         typename DType, typename DstPlan, typename Plan>
__global__ void MapRedKeepLowestKernel(DstPlan dst, Plan plan,
                                       DType scale, Shape<2> eshape) {
  const unsigned warp_size = 1 << warp_bits;
  const unsigned x = (blockIdx.x << warp_bits) + threadIdx.x;
  // to avoid bank conflict
  __shared__ DType s_res[warp_size][warp_size + 1];
  // note: reverse store [y][x], so that we can reduce over threadIdx.x, use warp optimization
  if (threadIdx.y < eshape[0] && x < eshape[1]) {
    s_res[threadIdx.x][threadIdx.y] = plan.Eval(threadIdx.y, x);
  }
  for (unsigned y = warp_size; y < eshape[0]; y += warp_size) {
    if (threadIdx.y + y < eshape[0] && x < eshape[1]) {
      Reducer::Reduce(s_res[threadIdx.x][threadIdx.y], plan.Eval(threadIdx.y + y, x));
    }
  }
  __syncthreads();
  if (eshape[0] >= warp_size) {
    Reduce1D<Reducer, warp_bits>(s_res[threadIdx.y]);
  } else {
    Reduce1DNotAlign<Reducer, warp_bits>(s_res[threadIdx.y], eshape[0]);
  }
  __syncthreads();

  if (threadIdx.y == 0 && x < eshape[1]) {
    Saver::Save(dst.REval(0, x),  s_res[threadIdx.x][0] * scale);
  }
}

template<typename Saver, typename Reducer,
         typename DstExp, typename E, typename DType>
inline void MapReduceKeepLowest(expr::Plan<DstExp, DType> dst,
                                const expr::Plan<E, DType> &plan,
                                DType scale, Shape<2> eshape,
                                cudaStream_t stream) {
  dim3 dimBlock(kMemUnit, kMemUnit);
  dim3 dimGrid((eshape[1] + kMemUnit - 1) >> kMemUnitBits);
  CheckLaunchParam(dimGrid, dimBlock, "MapRedKeepLowestKernel");
  MapRedKeepLowestKernel<Saver, Reducer, kMemUnitBits, DType,
                         expr::Plan<DstExp, DType>,
                         expr::Plan<E, DType> >
      <<<dimGrid, dimBlock, 0, stream>>>(dst, plan, scale, eshape);
}

template<typename Saver, typename Reducer, int block_dim_bits,
         typename DType, typename DstPlan, typename Plan>
__global__ void MapReduceKeepDim1Kernel(DstPlan dst, Plan plan, DType scale, Shape<4> pshape) {
  const int block_size = 1 << block_dim_bits;
  __shared__ DType s_rec[block_size];
  const int c = blockIdx.x;
  const index_t tot = pshape[3] * pshape[2] * pshape[0];

  DType res; Reducer::SetInitValue(res);
  for (index_t i_offset = 0; i_offset < tot; i_offset += block_size) {
    index_t i = i_offset + threadIdx.x;
    if (i< tot) {
      const index_t x = i % pshape[3];
      i /= pshape[3];
      const index_t y = i % pshape[2];
      const index_t n = i / pshape[2];
      Reducer::Reduce(res, plan.Eval((n * pshape[1] + c) * pshape[2] + y, x));
    }
  }
  s_rec[threadIdx.x] = res;
  __syncthreads();
  Reduce1D<Reducer, block_dim_bits>(s_rec);
  if (threadIdx.x == 0) {
    Saver::Save(dst.REval(0, c), s_rec[0] * scale);
  }
}

template<typename Saver, typename Reducer, typename DstExp, typename E, typename DType>
inline void MapReduceKeepDim1(expr::Plan<DstExp, DType> dst,
                              const expr::Plan<E, DType> &plan,
                              DType scale, Shape<4> pshape,
                              cudaStream_t stream) {
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid (pshape[1]);
  CheckLaunchParam(dimGrid, dimBlock, "MapReduceKeepDim1");
  MapReduceKeepDim1Kernel<Saver,Reducer,kBaseThreadBits, DType,
                          expr::Plan<DstExp, DType>,
                          expr::Plan<E, DType> >
      <<<dimGrid, dimBlock, 0, stream>>>(dst, plan, scale, pshape);
}

template<int x_bits, typename DType, typename DstPlan, typename SrcPlan1, typename SrcPlan2>
__global__ void SoftmaxGradKernel(DstPlan dst, SrcPlan1 src, SrcPlan2 label, index_t xmax) {
  const unsigned x_size = 1 << x_bits;
  const int y = blockIdx.x;
  const int k = static_cast<int>(label.Eval(0, y));

  // calculate normalizer, with writeback
  for (unsigned x = 0; x < xmax; x += x_size) {
    const unsigned xindex = x + threadIdx.x;
    if (xindex < xmax) {
      if (xindex == k) {
        dst.REval(y, xindex) = src.Eval(y, xindex) - 1.0f;
      } else {
        dst.REval(y, xindex) = src.Eval(y, xindex);
      }
    }
  }
}

template<int x_bits, typename DType,  typename DstPlan, typename SrcPlan>
__global__ void SoftmaxKernel(DstPlan dst, SrcPlan src, index_t xmax) {
  const unsigned x_size = 1 << x_bits;
  const int y = blockIdx.x;
  __shared__ DType s_rec[x_size];
  // step 1: get max
  if (threadIdx.x < xmax) {
    s_rec[threadIdx.x] = src.Eval(y, threadIdx.x);
  }
  for (unsigned x = x_size; x < xmax; x += x_size) {
    if (x + threadIdx.x < xmax) {
      DType a = src.Eval(y, x + threadIdx.x);
      s_rec[threadIdx.x] = max(a, s_rec[threadIdx.x]);
    }
  }
  __syncthreads();
  if (threadIdx.x >= xmax) {
    s_rec[threadIdx.x] = s_rec[0];
  }
  __syncthreads();
  Reduce1D<red::maximum, x_bits>(s_rec);
  __syncthreads();
  DType smax = s_rec[0];
  __syncthreads();
  s_rec[threadIdx.x] = 0.0f;
  __syncthreads();

  // calculate normalizer, with writeback
  for (unsigned x = 0; x < xmax; x += x_size) {
    if (x + threadIdx.x < xmax) {
      DType p = expf(src.Eval(y, x + threadIdx.x) - smax);
      s_rec[threadIdx.x] += p;
      // write back first, will fetch later
      dst.REval(y, x + threadIdx.x) = p;
    }
  }
  // calculate normalizer
  __syncthreads();
  Reduce1D<red::sum, x_bits>(s_rec);
  __syncthreads();
  DType ssum = s_rec[0];

  for (unsigned x = 0; x < xmax; x += x_size) {
    if (x + threadIdx.x < xmax) {
      dst.REval(y, x + threadIdx.x) /= ssum;
    }
  }
}

template<typename DType>
inline void Softmax(Tensor<gpu, 2, DType> &dst,
                    const Tensor<gpu, 2, DType> &src) {
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid(dst.size(0));
  CHECK_EQ(dst.shape_, src.shape_) << "Softmax: shape mismatch";
  CheckLaunchParam(dimGrid, dimBlock, "Softmax");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  SoftmaxKernel<kBaseThreadBits, DType>
      <<<dimGrid, dimBlock, 0, stream>>>
      (expr::MakePlan(dst),
       expr::MakePlan(src),
       dst.size(1));
}

template<typename DType>
inline void SoftmaxGrad(Tensor<gpu, 2, DType> &dst,
                        const Tensor<gpu, 2, DType> &src,
                        const Tensor<gpu, 1, DType> &label) {
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid(dst.size(0));
  CHECK_EQ(dst.shape_, src.shape_) << "SoftmaxGrad: shape mismatch";
  CHECK_EQ(dst.size(0), label.size(0)) << "SoftmaxGrad: label shape mismatch";
  CheckLaunchParam(dimGrid, dimBlock, "SoftmaxGrad");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  SoftmaxGradKernel<kBaseThreadBits, DType>
      <<<dimGrid, dimBlock, 0, stream>>>
      (expr::MakePlan(dst),
       expr::MakePlan(src),
       expr::MakePlan(label),
       dst.size(1));
}

template<int n_bits, typename DType>
__global__ void Softmax3DGradKernel(Tensor<gpu, 3, DType> dst,
                                    const Tensor<gpu, 3, DType> src,
                                    const Tensor<gpu, 2, DType> label) {
  const index_t xmax = dst.size(1);
  const index_t nmax = dst.size(2);
  const unsigned n_size = 1 << n_bits;
  const int y = blockIdx.x;
  const int n = threadIdx.x;

  for (index_t n_index = n; n_index < nmax; n_index += n_size) {
    const int k = static_cast<int>(label[y][n_index]);
    for (index_t i = 0; i < xmax; ++i) {
      if (i == k) {
        dst[y][i][n_index] = src[y][i][n_index] - 1.0f;
      } else {
        dst[y][i][n_index] = src[y][i][n_index];
      }
    }
  }
}

template<int n_bits, typename DType>
__global__ void Softmax3DGradKernel(Tensor<gpu, 3, DType> dst,
                                    const Tensor<gpu, 3, DType> src,
                                    const Tensor<gpu, 2, DType> label,
                                    DType ignore_label) {
  const index_t xmax = dst.size(1);
  const index_t nmax = dst.size(2);
  const unsigned n_size = 1 << n_bits;
  const int y = blockIdx.x;
  const int n = threadIdx.x;
  for (index_t n_index = n; n_index < nmax; n_index += n_size) {
    int k = static_cast<int>(label[y][n_index]);
    if (k == static_cast<int>(ignore_label)) {
      for (index_t i = 0; i < xmax; ++i) {
        dst[y][i][n_index] = 0.0f;
      }
    } else {
      for (index_t i = 0; i < xmax; ++i) {
        if (i == k) {
          dst[y][i][n_index] = src[y][i][n_index] - 1.0f;
        } else {
          dst[y][i][n_index] = src[y][i][n_index];
        }
      }
    }
  }
}

template<int n_bits, typename DType>
__global__ void Softmax3DKernel(Tensor<gpu, 3, DType> dst,
                    const Tensor<gpu, 3, DType> src) {
  const index_t xmax = dst.size(1);
  const index_t nmax = dst.size(2);
  const unsigned n_size = 1 << n_bits;
  const int y = blockIdx.x;
  const int n = threadIdx.x;

  for (index_t n_index = n; n_index < nmax; n_index += n_size) {
    DType smax = src[y][0][n_index];
    for (index_t i = 1; i < xmax; ++i) {
      smax = max(smax, src[y][i][n_index]);
    }
    DType ssum = 0.0f;
    for (index_t i = 0; i < xmax; ++i) {
      DType p = expf(src[y][i][n_index] - smax);
      ssum += p;
      dst[y][i][n_index] = p;
    }
    for (index_t i = 0; i < xmax; ++i) {
      dst[y][i][n_index] /= ssum;
    }
  }
}

template<typename DType>
inline void Softmax(Tensor<gpu, 3, DType> &dst,
                    const Tensor<gpu, 3, DType> &src) {
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid(dst.size(0));
  CHECK_EQ(dst.shape_, src.shape_) << "Softmax: shape mismatch";
  CheckLaunchParam(dimGrid, dimBlock, "Softmax");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  Softmax3DKernel<kBaseThreadBits, DType><<<dimGrid, dimBlock, 0, stream>>>(dst, src);
}

template<typename DType>
inline void SoftmaxGrad(Tensor<gpu, 3, DType> &dst,
                        const Tensor<gpu, 3, DType> &src,
                        const Tensor<gpu, 2, DType> &label) {
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid(dst.size(0));
  CHECK_EQ(dst.shape_, src.shape_) << "SoftmaxGrad: shape mismatch";
  CHECK_EQ(dst.size(0), label.size(0)) << "SoftmaxGrad: label shape mismatch";
  CHECK_EQ(dst.size(2), label.size(1)) << "SoftmaxGrad: label shape mismatch";
  CheckLaunchParam(dimGrid, dimBlock, "SoftmaxGrad");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  Softmax3DGradKernel<kBaseThreadBits, DType><<<dimGrid, dimBlock, 0, stream>>>(dst, src, label);
}

template<typename DType>
inline void SoftmaxGrad(Tensor<gpu, 3, DType> &dst,
                        const Tensor<gpu, 3, DType> &src,
                        const Tensor<gpu, 2, DType> &label,
                        const DType &ignore_label) {
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid(dst.size(0));
  CHECK_EQ(dst.shape_, src.shape_) << "SoftmaxGrad: shape mismatch";
  CHECK_EQ(dst.size(0), label.size(0)) << "SoftmaxGrad: label shape mismatch";
  CHECK_EQ(dst.size(2), label.size(1)) << "SoftmaxGrad: label shape mismatch";
  CheckLaunchParam(dimGrid, dimBlock, "SoftmaxGrad");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  Softmax3DGradKernel<kBaseThreadBits, DType><<<dimGrid, dimBlock, 0, stream>>>(dst, src, label, ignore_label);
}

#include <cmath>
#include <cfloat>

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  // Number of ROIs
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  int* argmax_data = max_idx_.mutable_cpu_data();
  caffe_set(top_count, -1, argmax_data);

  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = round(bottom_rois[1] * spatial_scale_);
    int roi_start_h = round(bottom_rois[2] * spatial_scale_);
    int roi_end_w = round(bottom_rois[3] * spatial_scale_);
    int roi_end_h = round(bottom_rois[4] * spatial_scale_);
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    const Dtype bin_size_h = static_cast<Dtype>(roi_height)
                             / static_cast<Dtype>(pooled_height_);
    const Dtype bin_size_w = static_cast<Dtype>(roi_width)
                             / static_cast<Dtype>(pooled_width_);

    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);

    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                              * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                              * bin_size_w));
          int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                           * bin_size_h));
          int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                           * bin_size_w));

          hstart = min(max(hstart + roi_start_h, 0), height_);
          hend = min(max(hend + roi_start_h, 0), height_);
          wstart = min(max(wstart + roi_start_w, 0), width_);
          wend = min(max(wend + roi_start_w, 0), width_);

          bool is_empty = (hend <= hstart) || (wend <= wstart);

          const int pool_index = ph * pooled_width_ + pw;
          if (is_empty) {
            top_data[pool_index] = 0;
            argmax_data[pool_index] = -1;
          }

          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width_ + w;
              if (batch_data[index] > top_data[pool_index]) {
                top_data[pool_index] = batch_data[index];
                argmax_data[pool_index] = index;
              }
            }
          }
        }
      }
      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      argmax_data += max_idx_.offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }
}

template<typename Dtype>
__global__ void ROIPoolForward(const int nthreads, const Dtype* bottom_data,
                               const Dtype spatial_scale, const int channels, const int height,
                               const int width, const int pooled_height, const int pooled_width,
                               const Dtype* bottom_rois, Dtype* top_data, int* argmax_data) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = std::round(bottom_rois[1] * spatial_scale);
    int roi_start_h = std::round(bottom_rois[2] * spatial_scale);
    int roi_end_w = std::round(bottom_rois[3] * spatial_scale);
    int roi_end_h = std::round(bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(pooled_width);

    int hstart = static_cast<int>(std::floor(static_cast<Dtype>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(std::floor(static_cast<Dtype>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(std::ceil(static_cast<Dtype>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(std::ceil(static_cast<Dtype>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = std::min(std::max(hstart + roi_start_h, 0), height);
    hend = std::min(std::max(hend + roi_start_h, 0), height);
    wstart = std::min(std::max(wstart + roi_start_w, 0), width);
    wend = std::min(std::max(wend + roi_start_w, 0), width);
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
    argmax_data[index] = maxidx;
  }
}

template <typename Dtype>
__global__ void ROIPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int num_rois, const Dtype spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* bottom_diff,
    const Dtype* bottom_rois) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x) {
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

      int roi_start_w = std::round(offset_bottom_rois[1] * spatial_scale);
      int roi_start_h = std::round(offset_bottom_rois[2] * spatial_scale);
      int roi_end_w = std::round(offset_bottom_rois[3] * spatial_scale);
      int roi_end_h = std::round(offset_bottom_rois[4] * spatial_scale);

      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                           h >= roi_start_h && h <= roi_end_h);
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
      const int* offset_argmax_data = argmax_data + offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);

      Dtype bin_size_h = static_cast<Dtype>(roi_height)
                         / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = static_cast<Dtype>(roi_width)
                         / static_cast<Dtype>(pooled_width);

      int phstart = std::floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
      int phend = std::ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
      int pwstart = std::floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
      int pwend = std::ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);

      phstart = std::min(std::max(phstart, 0), pooled_height);
      phend = std::min(std::max(phend, 0), pooled_height);
      pwstart = std::min(std::max(pwstart, 0), pooled_width);
      pwend = std::min(std::max(pwend, 0), pooled_width);

      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (offset_argmax_data[ph * pooled_width + pw] == (h * width + w)) {
            gradient += offset_top_diff[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

#include <vector>
#include <iostream>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 5];
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
    const float *cur_box = dev_boxes + cur_box_idx * 5;
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
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

void _nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh) {
  float* boxes_dev = NULL;
  unsigned long long* mask_dev = NULL;

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

  CUDA_CHECK(cudaMalloc(&boxes_dev,
                        boxes_num * boxes_dim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(boxes_dev,
                        boxes_host,
                        boxes_num * boxes_dim * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&mask_dev,
                        boxes_num * col_blocks * sizeof(unsigned long long)));

  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  CUDA_CHECK(cudaMemcpy(&mask_host[0],
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

  CUDA_CHECK(cudaFree(boxes_dev));
  CUDA_CHECK(cudaFree(mask_dev));
}

}  // namespace cuda
}  // namespace mshadow
#endif  // MSHADOW_CUDA_TENSOR_GPU_INL_CUH_
