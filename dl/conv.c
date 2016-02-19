#include "layer.h"

// convert bottom3d (C x H x W)
//         -> bottom5d (C x kernel_h x kernel_w x H5 x W5)
//   bottom5d[c][kh][kw][h5][w5] = bottom3d[c][h][w]
//     h = (-pad_h + stride_h * h5) + kh
//     w = (-pad_w + stride_w * w5) + kw
//     if !(0 <= h < H) or !(0 <= w < W), assign 0
void convert_bottom(const real* bottom3d, real* const bottom5d,
                    const int C, const int H, const int W,
                    const int H5, const int W5,
                    const int kernel_h, const int kernel_w,
                    const int pad_h, const int pad_w,
                    const int stride_h, const int stride_w)
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

void forward(const Tensor* bottom3d, Tensor* const top3d,
             const Tensor* weight4d, const Tensor* bias1d,
             real* const temp_data, const real* const_data,
             const ConvOption* options)
{
  // weight shape: C' x C x kernel_h x kernel_w
  const int top_C = weight4d->shape[0][0];  // C'
  const int bottom_C = weight4d->shape[0][1];  // C
  const int kernel_h = weight4d->shape[0][2];
  const int kernel_w = weight4d->shape[0][3];

  // padding size & stride size
  const int pad_h = options->pad_h;
  const int pad_w = options->pad_w;
  const int stride_h = options->stride_h;
  const int stride_w = options->stride_w;

  // do forward-pass for each item in the batch
  const real* p_bottom_data = bottom3d->data;
  real* p_top_data = top3d->data;
  const int num_items = bottom3d->num_items;
  for (int n = 0; n < num_items; ++n) {
    // bottom shape: C x H x W
    const int bottom_H = bottom3d->shape[n][1];  // H
    const int bottom_W = bottom3d->shape[n][2];  // W

    // set top shape: C' x H' x W'
    //   H' = 1 + (H + 2*pad_h - kernel_h) / stride_h
    //   W' = 1 + (W + 2*pad_w - kernel_w) / stride_w
    const int top_H = 1 + (bottom_H + 2 * pad_h - kernel_h) / stride_h;
    const int top_W = 1 + (bottom_W + 2 * pad_w - kernel_w) / stride_w;
    top3d->ndim = 3;
    top3d->num_items = num_items;
    top3d->shape[n][0] = top_C;
    top3d->shape[n][1] = top_H;
    top3d->shape[n][2] = top_W;

    // convert bottom shape: C x H x W -> (C * kernel_h * kernel_w) x (H' * W')
    convert_bottom(p_bottom_data, temp_data,
                   bottom_C, bottom_H, bottom_W, top_H, top_W,
                   kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w);

    // top = dot(weight, bottom)
    //   weight: C' x (C * kernel_h * kernel_w)
    //   bottom: (C * kernel_h * kernel_w) x (H' * W')
    //   top: C' x H' x W'
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                top_C, top_H * top_W, bottom_C * kernel_h * kernel_w,
                (real)1.0, weight4d->data, bottom_C * kernel_h * kernel_w,
                temp_data, top_H * top_W,
                (real)0.0, p_top_data, top_H * top_W);

    // top = top + bias
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                top_C, top_H * top_W, 1,
                (real)1.0, bias1d->data, 1,
                const_data, top_H * top_W,
                (real)1.0, p_top_data, top_H * top_W);

    // locate next data
    p_bottom_data += bottom_C * bottom_H * bottom_W;
    p_top_data += top_C * top_H * top_W;
  } // endfor batch
}

void backward(Tensor *top_grad, Tensor *bottom_grad, Tensor *top_layer, Tensor *bottom_layer, ConvOption *options)
{
  return;
}

int main(int argc, char **argv)
{
  Tensor X, Y, W, b;
  real X_data[5000], Y_data[5000], W_data[500], b_data[50], temp_data[5000], const_data[5000];
  ConvOption option;
 {
  X.ndim = 3; X.num_items = 2;
  for (int i = 0; i < X.num_items; ++i) {
    X.shape[i][0] = 10;
    X.shape[i][1] = 5;
    X.shape[i][2] = 5;
  }
  W.ndim = 4; W.num_items = 1; W.shape[0][0] = 5; W.shape[0][1] = 10; W.shape[0][2] = 3; W.shape[0][3] = 3;
  b.ndim = 1; b.num_items = 1; b.shape[0][0] = 5;
  X.data = &X_data[0];
  Y.data = &Y_data[0];
  W.data = &W_data[0];
  b.data = &b_data[0]; b_data[0] = 0.1; b_data[1] = -0.1; b_data[2] = 0.2; b_data[3] = -0.2; b_data[4] = 0;
  option.kernel_h = 3;
  option.kernel_w = 3;
  option.pad_h = 1;
  option.pad_w = 1;
  option.stride_h = 2;
  option.stride_w = 2;
 }
 {
  FILE* fp;
  fp = fopen("X.txt", "r");
  int X_size = flatten_size(&X);
  for (int i = 0; i < X_size; ++i)
    fscanf(fp, "%f", &X.data[i]);
  fclose(fp);
  fp = fopen("W.txt", "r");
  int W_size = flatten_size(&W);
  for (int i = 0; i < W_size; ++i)
    fscanf(fp, "%f", &W.data[i]);
  fclose(fp);
  for (int i = 0; i < 5000; ++i) {
    const_data[i] = 1;
  }
 }
 {
  real* p_Y_data = Y.data;
  forward(&X, &Y, &W, &b, temp_data, const_data, &option);
  for (int n = 0; n < Y.num_items; ++n) {
    printf("Y[%d] (%d x %d x %d)\n", n, Y.shape[n][0], Y.shape[n][1], Y.shape[n][2]);
    for (int c = 0; c < Y.shape[n][0]; ++c) {
      for (int h = 0; h < Y.shape[n][1]; ++h) {
        for (int w = 0; w < Y.shape[n][2]; ++w) {
          printf("%03.6f ", p_Y_data[(c * Y.shape[n][1] + h) * Y.shape[n][2] + w]);
        }
        printf("\n");
      }
      printf("\n\n");
    }
    p_Y_data += Y.shape[n][0] * Y.shape[n][1] * Y.shape[n][2];
    printf("\n\n===============================\n\n");
  }
 }
  return 0;
}
