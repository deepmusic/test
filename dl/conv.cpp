#include <cblas.h>
#include <string.h>
#include <stdio.h>

typedef unsigned short ushort;
typedef unsigned int uint;
typedef float real;

typedef struct Tensor_ {
  real* data;
  ushort ndim;
  ushort shape[5];
} Tensor;

typedef struct ConvOption_ {
  ushort kernel_h;
  ushort kernel_w;
  ushort pad_h;
  ushort pad_w;
  ushort stride_h;
  ushort stride_w;
} ConvOption;

inline void convert_bottom(const real* bottom3d, real* const bottom5d,
                           const ushort C, const ushort H, const ushort W,
                           const ushort H5, const ushort W5,
                           const ushort kernel_h, const ushort kernel_w,
                           const ushort pad_h, const ushort pad_w,
                           const ushort stride_h, const ushort stride_w)
{
  for (int c = 0; c < C; ++c) {
   for (int kh = 0; kh < kernel_h; ++kh) {
    for (int kw = 0; kw < kernel_w; ++kw) {
      // pointer to bottom5d[c][kh][kw][h5 = 0][w5 = 0]
      real* const p_bottom5d = bottom5d + ((c * kernel_h + kh) * kernel_w + kw) * H5 * W5;
      int h = -pad_h + kh;
      int h5 = 0;

      // for h < 0 (zero-padded region): bottom5d[c][kh][kw][h5][:] = 0
      for (; h < 0; h += stride_h, ++h5);
      memset(p_bottom5d, 0, h5 * W5 * sizeof(real));

      // for 0 <= h < H (data region)
      for (; h < H && h5 < H5; h += stride_h, ++h5) {
        // pointer to bottom3d[c][h][w = 0]
        int w = -pad_w + kw;
        int w5 = 0;

        // for w < 0 (zero-padded region): bottom5d[c][kh][kw][h5][w5] = 0
        for (; w < 0; w += stride_w, ++w5) {
          p_bottom5d[h5 * W5 + w5] = 0;
        }

        // for 0 <= w < W (data region):
        //   bottom5d[c][kh][kw][h5][w5] = bottom3d[c][h][w]
        for (; w < W && w5 < W5; w += stride_w, ++w5) {
          p_bottom5d[h5 * W5 + w5] = bottom3d[(c * H + h) * W + w];
        }

        // for w >= W (zero-padded region): bottom5d[c][kh][kw][h5][w5] = 0
        for (; w5 < W5; ++w5) {
          p_bottom5d[h5 * W5 + w5] = 0;
        }
      }

      // for h >= H (zero-padded region): bottom5d[c][kh][kw][h5][:] = 0
      memset(p_bottom5d + h5 * W5, 0, (H5 - h5) * W5 * sizeof(real));
    } // endfor kw
   } // endfor kh
  } // endfor c
}

void forward(const Tensor* bottom4d, Tensor* const top4d,
             const Tensor* weight4d, const Tensor* bias1d,
             real* const temp_data, const real* const_data,
             const ConvOption* options)
{
  // bottom shape: N x C x H x W
  const ushort num_batch = bottom4d->shape[0];  // N
  const ushort bottom_C = bottom4d->shape[1];  // C
  const ushort bottom_H = bottom4d->shape[2];  // H
  const ushort bottom_W = bottom4d->shape[3];  // W

  // weight shape: C' x C x kernel_h x kernel_w
  const ushort top_C = weight4d->shape[0];  // C'
  const ushort kernel_h = weight4d->shape[2];
  const ushort kernel_w = weight4d->shape[3];

  // padding size & stride size
  const ushort pad_h = options->pad_h;
  const ushort pad_w = options->pad_w;
  const ushort stride_h = options->stride_h;
  const ushort stride_w = options->stride_w;

  // set top shape: N x C' x H' x W'
  //   H' = 1 + (H + 2*pad_h - kernel_h) / stride_h
  //   W' = 1 + (W + 2*pad_w - kernel_w) / stride_w
  top4d->ndim = 4;
  top4d->shape[0] = num_batch;
  top4d->shape[1] = top_C;
  top4d->shape[2] = 1 + (bottom_H + 2 * pad_h - kernel_h) / stride_h;
  top4d->shape[3] = 1 + (bottom_W + 2 * pad_w - kernel_w) / stride_w;

 { // do forward-pass
  const ushort top_H = top4d->shape[2];
  const ushort top_W = top4d->shape[3];
  const uint bottom_CHW = (uint)bottom_C * bottom_H * bottom_W;
  const uint top_CHW = (uint)top_C * top_H * top_W;

  for (int n = 0; n < num_batch; ++n) {
    // locate n-th batch
    const real* p_bottom_data = bottom4d->data + n * bottom_CHW;
    real* p_top_data = top4d->data + n * top_CHW;

    // convert bottom shape: C x H x W -> (C * kernel_h * kernel_w) x (H' * W')
    convert_bottom(p_bottom_data, temp_data,
                   bottom_C, bottom_H, bottom_W, top_H, top_W,
                   kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w);

   { // do matrix operations
    const uint top_HW = (uint)top_H * top_W;
    const uint weight_col = (uint)bottom_C * kernel_h * kernel_w;

    // dot(weight, bottom) -> C' x H' x W'
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                top_C, top_HW, weight_col,
                (real)1.0, weight4d->data, weight_col,
                temp_data, top_HW,
                0, p_top_data, top_HW);

    // add bias
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                top_C, top_HW, 1,
                (real)1.0, bias1d->data, 1,
                const_data, top_HW,
                (real)1.0, p_top_data, top_HW);
   } // end matrix operations
  } // endfor batch
 } // end forward-pass
}

void backward(Tensor *top_grad, Tensor *bottom_grad, Tensor *top_layer, Tensor *bottom_layer, ConvOption *options)
{
  return;
}

uint flatten_size(const Tensor* tensor)
{
  uint size = 1;
  for (int d = 0; d < tensor->ndim; ++d)
    size *= tensor->shape[d];
  return size;
}

int main(int argc, char **argv)
{
  Tensor X, Y, W, b;
  real X_data[5000], Y_data[5000], W_data[500], b_data[50], temp_data[5000], const_data[5000];
  X.ndim = 4; X.shape[0] = 2; X.shape[1] = 10; X.shape[2] = 5; X.shape[3] = 5;
  W.ndim = 4; W.shape[0] = 5; W.shape[1] = 10; W.shape[2] = 3; W.shape[3] = 3;
  b.ndim = 1; b.shape[0] = 5;
  X.data = &X_data[0];
  Y.data = &Y_data[0];
  W.data = &W_data[0];
  b.data = &b_data[0]; b_data[0] = 0.1; b_data[1] = -0.1; b_data[2] = 0.2; b_data[3] = -0.2; b_data[4] = 0;
  ConvOption option;
  option.kernel_h = 3;
  option.kernel_w = 3;
  option.pad_h = 1;
  option.pad_w = 1;
  option.stride_h = 2;
  option.stride_w = 2;

  FILE* fp;
  fp = fopen("X.txt", "r");
  uint X_size = flatten_size(&X);
  for (int i = 0; i < X_size; ++i)
    fscanf(fp, "%f", &X.data[i]);
  fclose(fp);
  fp = fopen("W.txt", "r");
  uint W_size = flatten_size(&W);
  for (int i = 0; i < W_size; ++i)
    fscanf(fp, "%f", &W.data[i]);
  fclose(fp);
  for (int i = 0; i < 5000; ++i) {
    const_data[i] = 1;
  }

  forward(&X, &Y, &W, &b, temp_data, const_data, &option);
  printf("Y (%d x %d x %d x %d)\n", Y.shape[0], Y.shape[1], Y.shape[2], Y.shape[3]);
  for (int n = 0; n < Y.shape[0]; ++n) {
    for (int c = 0; c < Y.shape[1]; ++c) {
      for (int h = 0; h < Y.shape[2]; ++h) {
        for (int w = 0; w < Y.shape[3]; ++w) {
          printf("%03.6f ", Y.data[((n * Y.shape[1] + c) * Y.shape[2] + h) * Y.shape[3] + w]);
        }
        printf("\n");
      }
      printf("\n\n");
    }
    printf("\n\n===============================\n\n");
  }

  return 0;
}
