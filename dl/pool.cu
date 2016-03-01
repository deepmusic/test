#include "layer.h"

// --------------------------------------------------------------------------
// kernel code
//   max_pool_{gpu, cpu}
// --------------------------------------------------------------------------

// max-pooling bottom3d (C x H x W) -> top3d (C x H' x W')
//   given (c, h', w'),
//     top3d[c][h'][w'] = max_{h, w} bottom3d[c][h][w]
//     argmax3d[c][h'][w'] = argmax_{h, w} bottom3d[c][h][w]
//   for
//     h = (-pad_h + stride_h * h') + { 0, 1, ..., kernel_h - 1 }
//     w = (-pad_w + stride_w * w') + { 0, 1, ..., kernel_w - 1 }
#ifdef GPU
__global__
void max_pool_gpu(const real* const bottom3d,
                  real* const top3d,
                  int* const argmax3d,
                  const int C, const int bottom_H, const int bottom_W,
                  const int top_H, const int top_W,
                  const int kernel_h, const int kernel_w,
                  const int pad_h, const int pad_w,
                  const int stride_h, const int stride_w)
{
  // thread index: (c, h', w') = c*H'*W' + h'*W' + w'
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  {
    // parse thread index -> (c, h', w')
    const int c = index / top_H / top_W;
    const int ht = (index / top_W) % top_H;
    const int wt = index % top_W; 

    // pooling range in bottom
    //   h = (-pad_h + stride_h * h') + { 0, 1, ..., kernel_h - 1}
    //   w = (-pad_w + stride_w * w') + { 0, 1, ..., kernel_w - 1}
    const int h_start = MAX(0, -pad_h + stride_h * ht);
    const int w_start = MAX(0, -pad_w + stride_w * wt);
    const int h_end = MIN(bottom_H, -pad_h + stride_h * ht + kernel_h);
    const int w_end = MIN(bottom_W, -pad_w + stride_w * wt + kernel_w);

    // if pooling range is invalid, assign 0
    if (h_start >= h_end || w_start >= w_end) {
      top3d[index] = 0;
      argmax3d[index] = -1;
    }

    // otherwise,
    //   top3d[c][h'][w'] = max_{h,w} bottom3d[c][h][w]
    //   argmax3d[c][h'][w'] = argmax_{h,w} bottom3d[c][h][w]
    else {
      const real* const p_bottom3d = bottom3d + c * bottom_H * bottom_W;
      int maxidx = h_start * bottom_W + w_start;
      real maxval = p_bottom3d[maxidx];
      for (int h = h_start; h < h_end; ++h) {
        for (int w = w_start; w < w_end; ++w) {
          if (p_bottom3d[h * bottom_W + w] > maxval) {
            maxidx = h * bottom_W + w;
            maxval = p_bottom3d[maxidx];
          }
        }
      }
      top3d[index] = maxval;
      argmax3d[index] = maxidx;
    }
  }
}
#else
void max_pool_cpu(const real* const bottom3d,
                  real* const top3d,
                  int* const argmax3d,
                  const int C, const int bottom_H, const int bottom_W,
                  const int top_H, const int top_W,
                  const int kernel_h, const int kernel_w,
                  const int pad_h, const int pad_w,
                  const int stride_h, const int stride_w)
{
  // thread index: (c, h', w') = c*H'*W' + h'*W' + w'
  for (int index = 0; index < C * top_H * top_W; ++index) {
    // parse thread index -> (c, h', w')
    const int c = index / top_H / top_W;
    const int ht = (index / top_W) % top_H;
    const int wt = index % top_W; 

    // pooling range in bottom
    //   h = (-pad_h + stride_h * h') + { 0, 1, ..., kernel_h - 1}
    //   w = (-pad_w + stride_w * w') + { 0, 1, ..., kernel_w - 1}
    const int h_start = MAX(0, -pad_h + stride_h * ht);
    const int w_start = MAX(0, -pad_w + stride_w * wt);
    const int h_end = MIN(bottom_H, -pad_h + stride_h * ht + kernel_h);
    const int w_end = MIN(bottom_W, -pad_w + stride_w * wt + kernel_w);

    // if pooling range is invalid, assign 0
    if (h_start >= h_end || w_start >= w_end) {
      top3d[index] = 0;
      argmax3d[index] = -1;
    }

    // otherwise,
    //   top3d[c][h'][w'] = max_{h,w} bottom3d[c][h][w]
    //   argmax3d[c][h'][w'] = argmax_{h,w} bottom3d[c][h][w]
    else {
      const real* const p_bottom3d = bottom3d + c * bottom_H * bottom_W;
      int maxidx = h_start * bottom_W + w_start;
      real maxval = p_bottom3d[maxidx];
      for (int h = h_start; h < h_end; ++h) {
        for (int w = w_start; w < w_end; ++w) {
          if (p_bottom3d[h * bottom_W + w] > maxval) {
            maxidx = h * bottom_W + w;
            maxval = p_bottom3d[maxidx];
          }
        }
      }
      top3d[index] = maxval;
      argmax3d[index] = maxidx;
    }
  }
}
#endif



// --------------------------------------------------------------------------
// layer operator code
//   pool_forward
// --------------------------------------------------------------------------

// max-pooling: bottom -> top
//   bottom: C x H x W
//   top: C x H' x W'
//   argmax: C x H' x W' array
void pool_forward(const Tensor* const bottom3d,
                  Tensor* const top3d,
                  int* const argmax_data,
                  const PoolOption* const option)
{
  // kernel size, padding size & stride size
  const int kernel_h = option->kernel_h;
  const int kernel_w = option->kernel_w;
  const int pad_h = option->pad_h;
  const int pad_w = option->pad_w;
  const int stride_h = option->stride_h;
  const int stride_w = option->stride_w;

  // do forward-pass for each item in the batch
  const real* p_bottom_item = bottom3d->data;
  real* p_top_item = top3d->data;
  int* p_argmax_item = argmax_data;
  for (int n = 0; n < bottom3d->num_items; ++n) {
    // bottom shape: C x H x W
    const int C = bottom3d->shape[n][0];  // C
    const int bottom_H = bottom3d->shape[n][1];  // H
    const int bottom_W = bottom3d->shape[n][2];  // W

    // set top shape: C x H' x W'
    //   H' = 1 + (H + 2*pad_h - kernel_h) / stride_h
    //   W' = 1 + (W + 2*pad_w - kernel_w) / stride_w
    const int top_H
        = 1 + DIV_THEN_CEIL(bottom_H + 2 * pad_h - kernel_h, stride_h);
    const int top_W
        = 1 + DIV_THEN_CEIL(bottom_W + 2 * pad_w - kernel_w, stride_w);
    top3d->shape[n][0] = C;
    top3d->shape[n][1] = top_H;
    top3d->shape[n][2] = top_W;

    // max-pooling
    //   bottom3d (C x H x W) -> top3d (C x H' x W')
    #ifdef GPU
    {
      const int num_threads = C * top_H * top_W;
      const int threads_per_block = 512;
      const int num_blocks = DIV_THEN_CEIL(num_threads, threads_per_block);
      max_pool_gpu<<<num_blocks, threads_per_block>>>(
          p_bottom_item, p_top_item, p_argmax_item,
          C, bottom_H, bottom_W, top_H, top_W,
          kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w);
    }
    #else
    {
      max_pool_cpu(
          p_bottom_item, p_top_item, p_argmax_item,
          C, bottom_H, bottom_W, top_H, top_W,
          kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w);
    }
    #endif

    // locate next item
    {
      const int bottom_size = C * bottom_H * bottom_W;
      const int top_size = C * top_H * top_W;
      p_bottom_item += bottom_size;
      p_top_item += top_size;
      p_argmax_item += top_size;
    }
  } // endfor batch

  top3d->ndim = 3;
  top3d->num_items = bottom3d->num_items;
}



// --------------------------------------------------------------------------
// test code
// --------------------------------------------------------------------------

#ifdef TEST
#include <stdio.h>
#include <stdlib.h>

#define DATA_SIZE 128*72*92

int main(int argc, char *argv[])
{
  // variable declaration & memory allocation
  Tensor X, Y;
  real* const X_data = (real*)malloc(DATA_SIZE * sizeof(real));
  real* const Y_data = (real*)malloc(DATA_SIZE * sizeof(real));
  real* const Y_true_data = (real*)malloc(DATA_SIZE * sizeof(real));
  int* p_argmax_data;
  PoolOption option;

  // set option
  {
    option.kernel_h = 3;
    option.kernel_w = 3;
    option.pad_h = 0;
    option.pad_w = 0;
    option.stride_h = 2;
    option.stride_w = 2;
  }

  // set data shapes
  {
    X.ndim = 3;
    X.num_items = 1;
    for (int i = 0; i < X.num_items; ++i) {
      X.shape[i][0] = 128;
      X.shape[i][1] = 72;
      X.shape[i][2] = 92;
    }

    Y.ndim = 3;
    Y.num_items = 1;
    for (int i = 0; i < Y.num_items; ++i) {
      Y.shape[i][0] = X.shape[i][0];
      Y.shape[i][1] = 1 + DIV_THEN_CEIL(
          X.shape[i][1] + 2 * option.pad_h - option.kernel_h,
          option.stride_h);
      Y.shape[i][2] = 1 + DIV_THEN_CEIL(
          X.shape[i][2] + 2 * option.pad_w - option.kernel_w,
          option.stride_w);
    }
  }
 
  // load data
  {
    FILE* fp;
    const int X_size = flatten_size(&X);
    const int Y_size = flatten_size(&Y);

    printf("data loading\n");

    fp = fopen("../data/temp/pool_bottom0.bin", "rb");
    if ((int)fread(X_data, sizeof(real), X_size, fp) != X_size) {
      printf("Error while reading pool_bottom0\n");
    }
    fclose(fp);

    fp = fopen("../data/temp/pool_top0.bin", "rb");
    if ((int)fread(Y_true_data, sizeof(real), Y_size, fp) != Y_size) {
      printf("Error while reading pool_top0\n");
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

    printf("gpu malloc\n");
    CUDA_CHECK(cudaMalloc(&X.data, X_size * sizeof(real)));
    CUDA_CHECK(cudaMalloc(&Y.data, Y_size * sizeof(real)));
    CUDA_CHECK(cudaMalloc(&p_argmax_data, Y_size * sizeof(int)));

    printf("memcpy: cpu -> gpu\n");
    CUDA_CHECK(cudaMemcpy(X.data, X_data, X_size * sizeof(real),
                          cudaMemcpyHostToDevice));
  }
  #else
  {
    const int Y_size = flatten_size(&Y);

    X.data = X_data;
    Y.data = Y_data;
    p_argmax_data = (int*)malloc(Y_size * sizeof(int));
  }
  #endif

  // do forward operation
  {
    printf("do forward\n");
    pool_forward(&X, &Y, p_argmax_data, &option);
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
    int i = 0;

    printf("verification\n");

    for (int n = 0; n < Y.num_items; ++n) {
      for (int c = 0; c < Y.shape[n][0]; ++c) {
        for (int h = 0; h < Y.shape[n][1]; ++h) {
          for (int w = 0; w < Y.shape[n][2]; ++w) {
            if (Y_data[i] != Y_true_data[i]) {
              printf("Y[%d,%d,%d,%d] = %.6f  Y_true[%d,%d,%d,%d] = %.6f\n",
                     n, c, h, w, Y_data[i], n, c, h, w, Y_true_data[i]);
              ++i;
            }
          } // endfor w
        } // endfor h
      } // endfor c
    } // endfor n
  }

  // memory deallocation
  {
    printf("free\n");
    free(X_data);
    free(Y_data);
    free(Y_true_data);
  }
  #ifdef GPU
  {
    printf("gpu free\n");
    CUDA_CHECK(cudaFree(X.data));
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
