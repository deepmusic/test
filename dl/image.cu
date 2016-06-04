#include "layer.h"

#ifdef GPU
__global__
void bilinear_resize_gpu(const unsigned char* const img,
                         real* const input3d,
                         const int height, const int width,
                         const int resized_height, const int resized_width,
                         const real img_scale_y, const real img_scale_x)
{
  const real gs_mean[3] = { 102.9801f, 115.9465f, 122.7717f };

  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int resized_area = resized_height * resized_width;
  if (index < 3 * resized_area) {
    const int stride = width * 3;
    const int c = index / resized_area;
    const int i = (index / resized_width) % resized_height;
    const int j = index % resized_width;

    const real y = i / img_scale_y;
    const int y0 = (int)y;
    const int y1 = MIN(y0 + 1,  height - 1);
    const real ay = y - y0;
    const real by = 1 - ay;

    const real x = j / img_scale_x;
    const int x0 = (int)x;
    const int x1 = MIN(x0 + 1,  width - 1);
    const real ax = x - x0;
    const real bx = 1 - ax;

    real val = 0;
    val += (ax > 0 && ay > 0) ? ax * ay * img[y1 * stride + x1 * 3 + c] : 0;
    val += (ax > 0 && by > 0) ? ax * by * img[y0 * stride + x1 * 3 + c] : 0;
    val += (bx > 0 && ay > 0) ? bx * ay * img[y1 * stride + x0 * 3 + c] : 0;
    val += (bx > 0 && by > 0) ? bx * by * img[y0 * stride + x0 * 3 + c] : 0;

    input3d[index] = val - gs_mean[c];
  }
}
#else
void bilinear_resize_cpu(const unsigned char* const img,
                         real* const input3d,
                         const int height, const int width,
                         const int resized_height, const int resized_width,
                         const real img_scale_y, const real img_scale_x)
{
  static const real gs_mean_blue = 102.9801f;
  static const real gs_mean_green = 115.9465f;
  static const real gs_mean_red = 122.7717f;

  const int stride = width * 3;
  const int resized_area = resized_height * resized_width;
  real* const p_inputB = input3d + 0 * resized_area;
  real* const p_inputG = input3d + 1 * resized_area;
  real* const p_inputR = input3d + 2 * resized_area;

  for (int i = 0; i < resized_height; ++i) {
    const real y = i / img_scale_y;
    const int y0 = (int)y;
    const int y1 = MIN(y0 + 1,  height - 1);
    const real ay = y - y0;
    const real by = 1 - ay;
    for (int j = 0; j < resized_width; ++j) {
      const real x = j / img_scale_x;
      const int x0 = (int)x;
      const int x1 = MIN(x0 + 1,  width - 1);
      const real ax = x - x0;
      const real bx = 1 - ax;
      real B = 0, G = 0, R = 0;
      if (ax > 0 && ay > 0) {
        B += ax * ay * img[y1 * stride + x1 * 3 + 0];
        G += ax * ay * img[y1 * stride + x1 * 3 + 1];
        R += ax * ay * img[y1 * stride + x1 * 3 + 2];
      }
      if (ax > 0 && by > 0) {
        B += ax * by * img[y0 * stride + x1 * 3 + 0];
        G += ax * by * img[y0 * stride + x1 * 3 + 1];
        R += ax * by * img[y0 * stride + x1 * 3 + 2];
      }
      if (bx > 0 && ay > 0) {
        B += bx * ay * img[y1 * stride + x0 * 3 + 0];
        G += bx * ay * img[y1 * stride + x0 * 3 + 1];
        R += bx * ay * img[y1 * stride + x0 * 3 + 2];
      }
      if (bx > 0 && by > 0) {
        B += bx * by * img[y0 * stride + x0 * 3 + 0];
        G += bx * by * img[y0 * stride + x0 * 3 + 1];
        R += bx * by * img[y0 * stride + x0 * 3 + 2];
      }

      p_inputB[i * resized_width + j] = B - gs_mean_blue;
      p_inputG[i * resized_width + j] = G - gs_mean_green;
      p_inputR[i * resized_width + j] = R - gs_mean_red;
/*
      if (i == resized_height - 1) {
        printf("%d %d:  %d %d %d %d %f %f %f %f,  %d %d %d %d   %f %f %f,  %f %f %f\n", i, j, y0, y1, x0, x1, ay, by, ax, bx,
              img[y1 * stride + x1 * 3], img[y0 * stride + x1 * 3], img[y1 * stride + x0 * 3], img[y0 * stride + x0 * 3],
              B, G, R, p_inputB[i * resized_width + j], p_inputG[i * resized_width + j], p_inputR[i * resized_width + j]);
      }
*/
    }
  }
}
#endif

void img2input(const unsigned char* const img,
               Tensor* const input3d,
               Tensor* const img_info1d,
               unsigned char* const temp_data,
               const int height, const int width)
{
  static const real gs_max_size = 1000.0f;
  static const real gs_base_size = 600.0f;

  const int img_size_min = MIN(height,  width);
  const int img_size_max = MAX(height,  width);

  real img_scale = gs_base_size / img_size_min;
  if (ROUND(img_scale * img_size_max) > gs_max_size) {
    img_scale = gs_max_size / img_size_max;
  }

  const int gs_scale_base = 32;
  const real img_scale_y
      = (real)((int)(height * img_scale / gs_scale_base) * gs_scale_base)
        / height;
  const real img_scale_x
      = (real)((int)(width * img_scale / gs_scale_base) * gs_scale_base)
        / width;

  const int resized_height = ROUND(height * img_scale_y);
  const int resized_width = ROUND(width * img_scale_x);

  const int n = img_info1d->num_items;
  real* const p_img_info1d = img_info1d->data + n * 6;
  p_img_info1d[0] = (real)resized_height;
  p_img_info1d[1] = (real)resized_width;
  p_img_info1d[2] = img_scale_y;
  p_img_info1d[3] = img_scale_x;
  p_img_info1d[4] = (real)height;
  p_img_info1d[5] = (real)width;

  #ifdef GPU
  {
    const int num_threads = 3 * resized_height * resized_width;
    const int threads_per_block = 512;
    const int num_blocks = DIV_THEN_CEIL(num_threads,  threads_per_block);
    cudaMemcpyAsync(temp_data, img,
                    height * width * 3 * sizeof(unsigned char),
                    cudaMemcpyHostToDevice);
    bilinear_resize_gpu<<<num_blocks, threads_per_block>>>(
        temp_data,  input3d->data + input3d->start[n],
        height,  width,  resized_height,  resized_width,
        img_scale_y,  img_scale_x);
  }
  #else
  {
    bilinear_resize_cpu(
        img,  input3d->data + input3d->start[n],
        height,  width,  resized_height,  resized_width,
        img_scale_y,  img_scale_x);
  }
  #endif

  input3d->shape[n][0] = 3;
  input3d->shape[n][1] = resized_height;
  input3d->shape[n][2] = resized_width;
  ++input3d->num_items;

  #if BATCH_SIZE > 1
  if (n < BATCH_SIZE - 1) {
    input3d->start[n + 1] = input3d->start[n]
                            + 3 * resized_height * resized_width;
  }
  #endif

  img_info1d->shape[n][0] = 6;
  ++img_info1d->num_items;
}
