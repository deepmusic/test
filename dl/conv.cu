#include "layer.h"
#include "cuda_settings.h"

// TODO
#ifdef PASS
__global__ void convert_bottom_patch(const real* const bottom3d_patch,
                                     real* const bottom5d_hyperpatch,
                                     const int kernel_h, const int kernel_w,
                                     const int h_min, const int h_max,
                                     const int w_min, const int w_max,
                                     const int W, const int stride_hyperpatch)
{
  real* p_bottom5d = bottom5d_hyperpatch;
  for (int kh = blockIdx.x * blockDim.x + threadIdx.x;
       kh < kernel_h;
       kh += blockDim.x) {
    for (int kw = blockIdx.y * blockDim.y + threadIdx.y;
         kw < kernel_w;
         kw += blockDim.y) {
      const int index = kh * W + kw;
      if (kh >= h_min && kh < h_max && kw >= w_min && kw < w_max) {
        p_bottom5d[index * stride_hyperpatch] = bottom3d_patch[index];
      }
      else {
        p_bottom5d[index * stride_hyperpatch] = 0;
      }
    }
  }
}
#endif

// convert bottom3d (C x H x W)
//         -> bottom5d (C x kernel_h x kernel_w x H5 x W5)
//   given (c, h5, w5), for kh: [0, ..., kernel_h) and kw: [0, ..., kernel_w),
//     bottom5d[c][kh][kw][h5][w5] = bottom3d[c][h][w]
//       h = (-pad_h + stride_h * h5) + kh
//       w = (-pad_w + stride_w * w5) + kw
//       if !(0 <= h < H) or !(0 <= w < W), assign 0
__global__ void convert_bottom(const real* const bottom3d, real* const bottom5d,
                               const int C, const int H, const int W,
                               const int H5, const int W5,
                               const int kernel_h, const int kernel_w,
                               const int pad_h, const int pad_w,
                               const int stride_h, const int stride_w)
{
  const int H5W5 = H5 * W5;

  // thread index: (c, h5, w5) = c*H5*W5 + h5*W5 + w5
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < C * H5W5;
       index += blockDim.x) {
    // parse thread index -> (c, h5, w5)
    const int c = index / H5W5;
    const int h5 = (index / W5) % H5;
    const int w5 = index % W5; 
    // p_bottom5d initially points to bottom5d[c][kh = 0][kw = 0][h5][w5]
    real* p_bottom5d = bottom5d + index + (c * H5W5) * (kernel_h * kernel_w - 1);

    // (h_start, w_start): upper-left corner location of bottom3d's kernel patch
    const int h_start = h5 * stride_h - pad_h;
    const int w_start = w5 * stride_w - pad_w;
    const real* p_bottom3d = bottom3d + (c * H + h_start) * W + w_start;

#ifdef PASS
    dim3 num_threads(3, 3);
    dim3 num_blocks((kernel_h + 3 - 1)/3, (kernel_w + 3 - 1)/3);
    convert_bottom_patch<<<num_blocks, num_threads>>>(
        p_bottom3d, p_bottom5d,
        kernel_h, kernel_w,
        -h_start, H - h_start, -w_start, W - w_start,
        W, H5W5);
#else
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        if (h_start + kh >= 0 && h_start + kh < H &&
            w_start + kw >= 0 && w_start + kw < W) {
          // bottom5d[c][kh][kw][h5][w5] = bottom3d[c][h_start + kh][w_start + kw]
          p_bottom5d[(kh * kernel_w + kw) * H5W5] = p_bottom3d[kh * W + kw];
        }
        else {
          // if [h_start + kh][w_start + kw] is in a zero-padded region, assign 0
          p_bottom5d[(kh * kernel_w + kw) * H5W5] = 0;
        }
      }
    }
#endif
  }
}

// convolution: bottom -> top
//   bottom: (G * C) * H * W
//   top: (G * C') * H' * W'
//   weight: G x C' x C x kernel_h x kernel_w
//   bias: (G * C') x 1
//   temp: G * C * kernel_h * kernel_w * H' * W'
//   const: 1 x (G * C'), const[i] = 1 for all i
//   G: number of groups
void forward(const Tensor* const bottom3d, Tensor* const top3d,
             const Tensor* const weight5d, const Tensor* const bias1d,
             real* const temp_data, const real* const const_data,
             const ConvOption* const options)
{
  // weight shape: G x C' x C x kernel_h x kernel_w
  const int num_groups = weight5d->shape[0][0]; // G
  const int top_C = weight5d->shape[0][1];  // C'
  const int bottom_C = weight5d->shape[0][2];  // C
  const int kernel_h = weight5d->shape[0][3];
  const int kernel_w = weight5d->shape[0][4];

  // padding size & stride size
  const int pad_h = options->pad_h;
  const int pad_w = options->pad_w;
  const int stride_h = options->stride_h;
  const int stride_w = options->stride_w;

  // do forward-pass for each item in the batch
  const real* p_bottom_item = bottom3d->data;
  real* p_top_item = top3d->data;
  for (int n = 0; n < bottom3d->num_items; ++n) {
    // bottom shape: (G * C) x H x W
    const int bottom_H = bottom3d->shape[n][1];  // H
    const int bottom_W = bottom3d->shape[n][2];  // W

    // set top shape: (G * C') x H' x W'
    //   H' = 1 + (H + 2*pad_h - kernel_h) / stride_h
    //   W' = 1 + (W + 2*pad_w - kernel_w) / stride_w
    const int top_H = 1 + (bottom_H + 2 * pad_h - kernel_h) / stride_h;
    const int top_W = 1 + (bottom_W + 2 * pad_w - kernel_w) / stride_w;
    top3d->shape[n][0] = num_groups * top_C;
    top3d->shape[n][1] = top_H;
    top3d->shape[n][2] = top_W;

    // convert bottom shape
    //   (G * C) x H x W -> (G * C * kernel_h * kernel_w) x (H' * W')
    {
      // one thread computes "kernel_h * kernel_w" entries in top
      const int num_threads = num_groups * bottom_C * top_H * top_W;
      const int threads_per_block = 512;
      const int num_blocks = DIV_THEN_CEIL(num_threads, threads_per_block);
      convert_bottom<<<num_blocks, threads_per_block>>>(
          p_bottom_item, temp_data,
          num_groups * bottom_C, bottom_H, bottom_W,
          top_H, top_W,
          kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w);
    }

    // compute top[g] = dot(weight[g], bottom[g])
    //   weight[g]: C' x (C * kernel_h * kernel_w)
    //   bottom[g]: (C * kernel_h * kernel_w) x (H' * W')
    //   top[g]: C' x H' x W'
    for (int g = 0; g < num_groups; ++g) {
      const int kernel_size = bottom_C * kernel_h * kernel_w;
      const int top_area = top_H * top_W;
      const real* const p_temp_g = temp_data + g * kernel_size * top_area;
      const real* const p_weight_g = weight5d->data + g * top_C * kernel_size;
      real* const p_top_g = p_top_item + g * top_C * top_area;

      const cublasHandle_t* cublas_handle = (cublasHandle_t*)options->handle;
      const real one = 1.0, zero = 0.0;

      // compute Z = alpha * dot(X, Y) + beta * Z
      //   X (= weight): m x p,  Y (= bottom): p x n,  Z (= top): m x n
      //   X, Y, Z: row-major order (e.g., Z[i][j] = Z[i * n + j])
      // input arguments:
      //   &handle,
      //   do_transpose_Y (= false),  do_transpose_X (= false),
      //   n (= H' * W'),  m (= C'),  p (= C * kernel_h * kernel_w),
      //   &alpha (= 1),
      //   &Y,  number of columns in Y (= n),
      //   &X,  number of columns in X (= p),
      //   &beta (= 0),
      //   &Z,  number of columns in Z (= n)
      cublasSgemm(*cublas_handle,
                  CUBLAS_OP_N,  CUBLAS_OP_N,
                  top_area,  top_C,  kernel_size,
                  &one,
                  p_temp_g,  top_area,
                  p_weight_g,  kernel_size,
                  &zero,
                  p_top_g,  top_area);
    }

    // compute top[i][j] = top[i][j] + bias[i]
    //   top: (G * C') x (H' * W')
    //   bias: (G * C') x 1
    if (option->bias) {
      const int top_channels = num_groups * top_C;
      const int top_area = top_H * top_W;

      const cublasHandle_t* cublas_handle = (cublasHandle_t*)options->handle;
      const real one = 1.0;

      // the computation is equivalent to...
      //   top = top + dot(bias, constant)
      //   constant: 1 x (H' * W'), constant[i] = 1 for all i
      // thus, input arguments:
      //   do_transpose_Y (= false),  do_transpose_X (= false),
      //   n = H' * W',  m = G * C',  p = 1
      //   alpha = 1,  beta = 1
      cublasSgemm(*cublas_handle,
                  CUBLAS_OP_N,  CUBLAS_OP_N,
                  top_area,  top_channels,  1,
                  &one,
                  const_data,  top_area,
                  bias1d->data,  1,
                  &one,
                  p_top_item,  top_area);
    }

    // locate next item
    {
      const int bottom_size = num_groups * bottom_C * bottom_H * bottom_W;
      const int top_size = num_groups * top_C * top_H * top_W;
      p_bottom_item += bottom_size;
      p_top_item += top_size;
    }
  } // endfor batch

  top3d->ndim = 3;
  top3d->num_items = bottom3d->num_items;
}

// TODO
void backward(Tensor *top_grad, Tensor *bottom_grad, Tensor *top_layer, Tensor *bottom_layer, ConvOption *options)
{
  return;
}

#define DATA_SIZE 30000
#define WEIGHT_SIZE 100000
#define BIAS_SIZE 200

int main(int argc, char **argv)
{
  Tensor X, Y, W, b;
  real* X_data = (real*)malloc(DATA_SIZE * sizeof(real));
  real* Y_data = (real*)malloc(DATA_SIZE * sizeof(real));
  real* W_data = (real*)malloc(WEIGHT_SIZE * sizeof(real));
  real* b_data = (real*)malloc(BIAS_SIZE * sizeof(real));
  real* const_data = (real*)malloc(BIAS_SIZE * sizeof(real));
  real* p_temp_data;
  real* p_const_data;
  ConvOption option;
  cublasHandle_t cublas_handle;

  {
    option.num_groups = 1;
    option.out_channels = 100;
    option.kernel_h = 3;
    option.kernel_w = 3;
    option.pad_h = 1;
    option.pad_w = 1;
    option.stride_h = 1;
    option.stride_w = 1;
    option.bias = 1;
  }

  {
    X.ndim = 3;
    X.num_items = 10;
    for (int i = 0; i < X.num_items; ++i) {
      X.shape[i][0] = 100;
      X.shape[i][1] = 5;
      X.shape[i][2] = 5;
    }

    W.ndim = 5; W.num_items = 1;
    W.shape[0][0] = option.num_groups;
    W.shape[0][1] = option.out_channels / option.num_groups;
    W.shape[0][2] = X.shape[0][0] / option.num_groups;
    W.shape[0][3] = option.kernel_h;
    W.shape[0][4] = option.kernel_w;

    b.ndim = 1; b.num_items = 1;
    b.shape[0][0] = option.out_channels;
  }
 
  {
    printf("set device\n");
    CUDA_CHECK(cudaSetDevice(1));
    //printf("get device\n");
    //CUDA_CHECK(cudaGetDevice(0));
    printf("cublas initialization\n");
    if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
      printf("cublas creation failed\n");
    }
    option.handle = &cublas_handle;
  }

 {
  printf("cuda malloc\n");
  CUDA_CHECK(cudaMalloc(&X.data, DATA_SIZE*sizeof(real)));
  CUDA_CHECK(cudaMalloc(&Y.data, DATA_SIZE*sizeof(real)));
  CUDA_CHECK(cudaMalloc(&W.data, WEIGHT_SIZE*sizeof(real)));
  CUDA_CHECK(cudaMalloc(&b.data, BIAS_SIZE*sizeof(real)));
  CUDA_CHECK(cudaMalloc(&p_temp_data, DATA_SIZE*sizeof(real)));
  CUDA_CHECK(cudaMalloc(&p_const_data, DATA_SIZE*sizeof(real)));
 }
 {
  FILE* fp;
  int X_size = flatten_size(&X);
  int W_size = flatten_size(&W);
  int b_size = flatten_size(&b);
  printf("data loading\n");
  fp = fopen("X.txt", "r");
  for (int i = 0; i < X_size; ++i)
    fscanf(fp, "%f", &X_data[i]);
  fclose(fp);
  fp = fopen("W.txt", "r");
  for (int i = 0; i < W_size; ++i)
    fscanf(fp, "%f", &W_data[i]);
  fclose(fp);
  fp = fopen("b.txt", "r");
  for (int i = 0; i < b_size; ++i)
    fscanf(fp, "%f", &b_data[i]);
  fclose(fp);
  for (int i = 0; i < DATA_SIZE; ++i) {
    temp_data[i] = 1;
  }
 }
 {
  printf("memcopy\n");
  CUDA_CHECK(cudaMemcpy(X.data, X_data, DATA_SIZE*sizeof(real), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(W.data, W_data, WEIGHT_SIZE*sizeof(real), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b.data, b_data, BIAS_SIZE*sizeof(real), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(p_const_data, temp_data, DATA_SIZE*sizeof(real), cudaMemcpyHostToDevice));
 }
 {
  real* p_Y_data = &Y_data[0];

  printf("do forward\n");
  for (int i = 0; i < 100; ++i) {
    forward(&X, &Y, &W, &b, p_temp_data, p_const_data, &option);
    forward(&Y, &X, &W, &b, p_temp_data, p_const_data, &option);
  }

  printf("memcpy\n");
  CUDA_CHECK(cudaMemcpy(Y_data, X.data, DATA_SIZE*sizeof(real), cudaMemcpyDeviceToHost));

  for (int n = 0; n < Y.num_items; ++n) {
    printf("Y[%d] (%d x %d x %d)\n", n, Y.shape[n][0], Y.shape[n][1], Y.shape[n][2]);
    for (int c = 0; c < Y.shape[n][0]; ++c) {
      for (int h = 0; h < Y.shape[n][1]; ++h) {
        for (int w = 0; w < Y.shape[n][2]; ++w) {
          printf("%03.5f ", p_Y_data[(c * Y.shape[n][1] + h) * Y.shape[n][2] + w]);
        }
        printf("\n");
      }
      printf("\n\n");
    }
    p_Y_data += Y.shape[n][0] * Y.shape[n][1] * Y.shape[n][2];
    printf("\n\n===============================\n\n");
  }
 }
 {
  printf("cuda free\n");
  CUDA_CHECK(cudaFree(X.data));
  CUDA_CHECK(cudaFree(Y.data));
  CUDA_CHECK(cudaFree(W.data));
  CUDA_CHECK(cudaFree(b.data));
  CUDA_CHECK(cudaFree(p_temp_data));
  CUDA_CHECK(cudaFree(p_const_data));
  printf("cublas finalization\n");
  if (cublasDestroy(cublas_handle) != CUBLAS_STATUS_SUCCESS) {
    printf("cublas destruction failed\n");
  }
 }
  return 0;
}
