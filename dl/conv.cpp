typedef unsigned short ushort;
typedef unsigned int uint;
typedef float real;

void forward(const Tensor* bottom4d, Tensor* const top4d, const Tensor* weight4d, const Tensor* bias1d, real* const temp_data, const ConvOption* options)
{
  const ushort num_batch = bottom4d->shape[0];
  const ushort bottom_C = bottom4d->shape[1];
  const ushort bottom_H = bottom4d->shape[2];
  const ushort bottom_W = bottom4d->shape[3];
  const ushort top_C = weight4d->shape[0];
  const ushort kernel_h = weight4d->shape[2];
  const ushort kernel_w = weight4d->shape[3];
  const ushort pad_h = options->pad_h;
  const ushort pad_w = options->pad_w;
  const ushort stride_h = options->stride_h;
  const ushort stride_w = options->stride_w;

  /*
    set top shape: N x C' x H' x W'
      bottom shape: N x C x H x W
      weight shape: C' x C x kernel_h x kernel_w
      H' = 1 + (H + 2*pad_h - kernel_h) / stride_h
      W' = 1 + (W + 2*pad_w - kernel_w) / stride_w
  */
  top4d->ndim = 4;
  top4d->shape[0] = num_batch;
  top4d->shape[1] = top_C;
  top4d->shape[2] = 1 + (bottom_H + 2 * pad_h - kernel_h) / stride_h;
  top4d->shape[3] = 1 + (bottom_W + 2 * pad_w - kernel_w) / stride_w;

  // do forward-pass
  {
    const real* bottom_data = bottom4d->data;
    real* top_data = top4d->data;
    const ushort top_H = top4d->shape[2];
    const ushort top_W = top4d->shape[3];
    const uint bottom_CHW = bottom_C * bottom_H * bottom_W;
    const uint bottom_HW = bottom_H * bottom_W;
    const uint top_CHW = top_C * top_H * top_W;
    const uint top_HW = top_H * top_W;
    const uint num_patch = bottom_C * kernel_H * kernel_W;

    for (int n = 0; n < num_batch; ++n) {
      // locate n-th batch
      bottom_data += bottom_CHW;
      top_data += top_CHW;

      // convert bottom shape: C x H x W -> (C * kernel_h * kernel_w) x (H' * W')
      {
        int p_c = 0;
        for (int p = 0; p < num_patch; ++p) {
          int p_h = -pad_h;
          int h = 0;
          for (; p_h < 0; ++h) {
            for (int w = 0; w < top_W; ++w) {
              temp_data[p * top_HW + h * top_W + w] = 0;
            }
            p_h += stride_h;
          }
          for (; p_h < bottom_H; ++h) {
            int p_w = -pad_w;
            int w = 0;
            for (; p_w < 0; ++w) {
              temp_data[p * top_HW + h * top_W + w] = 0;
              p_w += stride_w;
            }
            for (; w < top_W; ++w) {
              temp_data[p * top_HW + h * top_W + w] =
                  bottom_data[p_c * bottom_HW + p_h * bottom_W + p_w];
              p_w += stride_w;
            }
            for (; w < top_W; ++w) {
              temp_data[p * top_HW + h * top_W + w] = 0;
            }
            p_h += stride_h;
          }
          for (; h < top_H; ++h) {
            for (ushort w = 0; w < top_W; ++w) {
              temp_data[p * top_HW + h * top_W + w] = 0;
            }
          }
          ++p_c;
        }
      }

      // dot(weight, bottom) -> C' x H' x W'

      // add bias
    }
  }
}

void backward(Tensor *top_grad, Tensor *bottom_grad, Tensor *top_layer, Tensor *bottom_layer, ConvOption *options)
{
}
