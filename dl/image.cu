#include "layer.h"
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>

using namespace cv;

void img2input(const unsigned char* const img,
               Tensor* const input3d,
               Tensor* const img_info1d,
               const int height, const int width, const int stride)
{
  static const real gs_max_size = 1000.0f;
  static const real gs_base_size = 600.0f;
  static const real gs_mean_blue = 102.9801f;
  static const real gs_mean_green = 115.9465f;
  static const real gs_mean_red = 122.7717f;

  const int img_size_min = MIN(height,  width);
  const int img_size_max = MAX(height,  width);

  real img_scale = gs_base_size / img_size_min;
  if (ROUND(img_scale * img_size_max) > gs_max_size) {
    img_scale = gs_max_size / img_size_max;
  }

  static const int gs_scale_base = 32;
  const real img_scale_x = (real)((int)(width * img_scale / gs_scale_base) * gs_scale_base) / width;
  const real img_scale_y = (real)((int)(height * img_scale / gs_scale_base) * gs_scale_base) / height;

  const int resized_height = ROUND(height * img_scale_y);
  const int resized_width = ROUND(width * img_scale_x);
  const int input_area = resized_height * resized_width;

  const int n = img_info1d->num_items;
  real* const p_img_info1d = img_info1d->data + n * 4;
  p_img_info1d[0] = resized_height;
  p_img_info1d[1] = resized_width;
  p_img_info1d[2] = img_scale_x;
  p_img_info1d[3] = img_scale_y;

  printf("start index = %d\n", input3d->start[n]);
  real* const p_inputB = input3d->data + input3d->start[n] + 0 * input_area;
  real* const p_inputG = input3d->data + input3d->start[n] + 1 * input_area;
  real* const p_inputR = input3d->data + input3d->start[n] + 2 * input_area;

  for (int i = 0; i < resized_height; ++i) {
    const real y = i / img_scale_y;
    const int y0 = (int)y;
    const int y1 = y0 + 1;
    const real ay = y - y0;
    const real by = 1 - ay;
    for (int j = 0; j < resized_width; ++j) {
      const real x = j / img_scale_x;		
      const int x0 = (int)x;			
      const int x1 = x0 + 1;
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
    }
  }

  const int input_size = 3 * resized_height * resized_width;
  printf("image size = %d x %d x 3 = %d\n", resized_height, resized_width, input_size);
  input3d->shape[n][0] = 3;
  input3d->shape[n][1] = resized_height;
  input3d->shape[n][2] = resized_width;
  input3d->start[n + 1] = input3d->start[n] + input_size;
  ++input3d->num_items;

  img_info1d->shape[n][0] = 4;
  ++img_info1d->num_items;
}

void load_image(const char* const filename,
                Tensor* const input3d,
                Tensor* const img_info1d)
{
  Mat image = imread(filename);
  if (!image.data) {
    printf("[ERROR] Cannot open image: %s\n", image.data);
  }

  const int height = image.rows;
  const int width = image.cols;
  const int stride = image.step.p[0];
  printf("Image %s: %d x %d, stride=%d\n", filename, height, width, stride);

  img2input(image.data, input3d, img_info1d, height, width, stride);
}
