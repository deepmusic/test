#include "core/net.h"
#include <string.h>

// --------------------------------------------------------------------------
// kernel code
//   bilinear_resize_{gpu, cpu}
//   compute_scaled_size
// --------------------------------------------------------------------------

#ifdef GPU
__global__
static
void bilinear_resize_gpu(const unsigned char img[],
                         real input3d[],
                         const int raw_h, const int raw_w,
                         const int resized_h, const int resized_w,
                         const real scale_h, const real scale_w)
{
  const real gs_mean[3] = { 102.9801f, 115.9465f, 122.7717f };

  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int resized_area = resized_h * resized_w;
  if (index < 3 * resized_area) {
    const int stride = raw_w * 3;
    const int c = index / resized_area;
    const int i = (index / resized_w) % resized_h;
    const int j = index % resized_w;

    const real y = i / scale_h;
    const int y0 = (int)y;
    const int y1 = MIN(y0 + 1,  raw_h - 1);
    const real ay = y - y0;
    const real by = 1 - ay;

    const real x = j / scale_w;
    const int x0 = (int)x;
    const int x1 = MIN(x0 + 1,  raw_w - 1);
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
static
void bilinear_resize_cpu(const unsigned char img[],
                         real input3d[],
                         const int raw_h, const int raw_w,
                         const int resized_h, const int resized_w,
                         const real scale_h, const real scale_w)
{
  static const real gs_mean_blue = 102.9801f;
  static const real gs_mean_green = 115.9465f;
  static const real gs_mean_red = 122.7717f;

  const int stride = raw_w * 3;
  const int resized_area = resized_h * resized_w;
  real* const p_inputB = input3d + 0 * resized_area;
  real* const p_inputG = input3d + 1 * resized_area;
  real* const p_inputR = input3d + 2 * resized_area;

  for (int i = 0; i < resized_h; ++i) {
    const real y = i / scale_h;
    const int y0 = (int)y;
    const int y1 = MIN(y0 + 1,  raw_h - 1);
    const real ay = y - y0;
    const real by = 1 - ay;
    for (int j = 0; j < resized_w; ++j) {
      const real x = j / scale_w;
      const int x0 = (int)x;
      const int x1 = MIN(x0 + 1,  raw_w - 1);
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

      p_inputB[i * resized_w + j] = B - gs_mean_blue;
      p_inputG[i * resized_w + j] = G - gs_mean_green;
      p_inputR[i * resized_w + j] = R - gs_mean_red;
    }
  }
}
#endif

static
void compute_scaled_size(const int raw_h, const int raw_w,
                         const int input_size, const int unit_size,
                         real* const p_scale_h, real* const p_scale_w,
                         int* const p_resized_h, int* const p_resized_w)
{
  real scale;

  // compute maximum scale factor satisfying both conditions:
  //   1. scale * "shorter side of raw image" <= short_side_cap
  //   2. scale * "longer side of raw image" <= long_side_cap
  // the cap is computed based on input_size:
  //   short_side_cap = input_size,  long_side_cap = 10/6 * input_size
  {
    const real long_side_cap = (real)(input_size * 10 / 6);
    const real short_side_cap = (real)input_size;
    const int short_side = MIN(raw_h,  raw_w);
    const int long_side = MAX(raw_h,  raw_w);

    scale = short_side_cap / short_side;
    if (ROUND(scale * long_side) > long_side_cap) {
      scale = long_side_cap / long_side;
    }
  }

  // calibrate scale factor such that
  // resized height & weight are multiples of unit_size
  *p_scale_h = (real)((int)(raw_h * scale / unit_size) * unit_size) / raw_h;
  *p_scale_w = (real)((int)(raw_w * scale / unit_size) * unit_size) / raw_w;
  *p_resized_h = (int)ROUND(raw_h * (*p_scale_h));
  *p_resized_w = (int)ROUND(raw_w * (*p_scale_w));

  #ifdef DEBUG
  printf("%d x %d --> %d x %d\n", raw_h, raw_w, *p_resized_h, *p_resized_w);
  #endif
}



// --------------------------------------------------------------------------
// output shape calculator code
// --------------------------------------------------------------------------

static
void image_shape(Tensor* const input3d,
                 Tensor* const img_info1d,
                 const int image_heights[],
                 const int image_widths[],
                 const int num_images,
                 long int* const p_temp_space,
                 const LayerOption* const option)
{
  // if no image is given (e.g., for initialization),
  // compute largest possible size based on input_size
  if (num_images == 0) {
    const int size = 3 * option->input_size * (option->input_size * 10 / 6);
    input3d->ndim = 3;
    input3d->num_items = BATCH_SIZE;
    for (int n = 0; n < input3d->num_items; ++n) {
      input3d->shape[n][0] = 3;
      input3d->shape[n][1] = option->input_size;
      input3d->shape[n][2] = option->input_size * 10 / 6;
      input3d->start[n] = n * size;
    }

    img_info1d->ndim = 1;
    img_info1d->num_items = BATCH_SIZE;
    for (int n = 0; n < img_info1d->num_items; ++n) {
      img_info1d->shape[n][0] = 6;
      img_info1d->start[n] = n * 6;
    }
  }

  // otherwise, compute scaled input size based on input_size
  else if (num_images <= BATCH_SIZE) {
    real scale_h, scale_w;
    int resized_h, resized_w;
    int total_size = 0;

    for (int n = 0; n < num_images; ++n) {
      compute_scaled_size(image_heights[n], image_widths[n],
                          option->input_size, option->unit_size,
                          &scale_h, &scale_w, &resized_h, &resized_w);

      input3d->shape[n][0] = 3;
      input3d->shape[n][1] = resized_h;
      input3d->shape[n][2] = resized_w;
      input3d->start[n] = total_size;
      total_size += 3 * resized_h * resized_w;

      img_info1d->shape[n][0] = 6;
      img_info1d->start[n] = n * 6;
    }

    input3d->ndim = 3;
    input3d->num_items = num_images;
    img_info1d->ndim = 1;
    img_info1d->num_items = num_images;
  }

  else {
    printf("[ERROR] Number of input images %d exceeds batch size %d\n",
           num_images, BATCH_SIZE);
  }

  // temporary data size: 3 * max_image_size^2
  //   used in GPU mode only, for copying raw image to GPU memory
  //   if max_image_size = 2048, GPU mode can take images <= 2048 x 2048
  #ifdef GPU
  *p_temp_space = 3 * option->max_image_size * option->max_image_size
                  * sizeof(unsigned char);
  #else
  *p_temp_space = 0;
  #endif
}



// --------------------------------------------------------------------------
// layer-wise operator code
// --------------------------------------------------------------------------

static
void image_forward(Tensor* const input3d,
                   Tensor* const img_info1d,
                   const unsigned char* const p_images[],
                   const int image_heights[],
                   const int image_widths[],
                   const int num_images,
                   unsigned char temp_data[],
                   const LayerOption* const option)
{
  #ifdef GPU
  real img_info[BATCH_SIZE * 6];
  #else
  real* img_info = img_info1d->data;
  #endif

  if (num_images == 0) {
    printf("[ERROR] No input image is given\n");
    return;
  }
  else if (num_images > BATCH_SIZE) {
    printf("[ERROR] Number of input images %d exceeds batch size %d\n",
           num_images, BATCH_SIZE);
    return;
  }

  for (int n = 0; n < num_images; ++n) {
    const int raw_h = image_heights[n];
    const int raw_w = image_widths[n];
    const int resized_h = input3d->shape[n][1];
    const int resized_w = input3d->shape[n][2];
    const real scale_h = (real)resized_h / raw_h;
    const real scale_w = (real)resized_w / raw_w;

    // check raw image size limit in GPU mode
    #ifdef GPU
    if (raw_h * raw_w > option->max_image_size * option->max_image_size) {
      printf("[ERROR] Input image size %d x %d exceeds limits %d x %d\n",
             raw_h, raw_w, option->max_image_size, option->max_image_size);
    }
    #endif

    // store image info
    img_info[n * 6 + 0] = (real)resized_h;
    img_info[n * 6 + 1] = (real)resized_w;
    img_info[n * 6 + 2] = scale_h;
    img_info[n * 6 + 3] = scale_w;
    img_info[n * 6 + 4] = (real)raw_h;
    img_info[n * 6 + 5] = (real)raw_w;

    // resize input image
    #ifdef GPU
    {
      const int num_threads = 3 * resized_h * resized_w;
      const int threads_per_block = 512;
      const int num_blocks = DIV_THEN_CEIL(num_threads,  threads_per_block);
      cudaMemcpyAsync(temp_data, p_images[n],
                      raw_h * raw_w * 3 * sizeof(unsigned char),
                      cudaMemcpyHostToDevice);
      bilinear_resize_gpu<<<num_blocks, threads_per_block>>>(
          temp_data,  input3d->data + input3d->start[n],
          raw_h,  raw_w,  resized_h,  resized_w,  scale_h,  scale_w);
    }
    #else
    {
      bilinear_resize_cpu(
          p_images[n],  input3d->data + input3d->start[n],
          raw_h,  raw_w,  resized_h,  resized_w,  scale_h,  scale_w);
    }
    #endif
  }

  // copy image info: main memory -> GPU memory
  #ifdef GPU
  cudaMemcpyAsync(img_info1d->data, img_info, 6 * num_images * sizeof(real),
                  cudaMemcpyHostToDevice);
  #endif
}



// --------------------------------------------------------------------------
// functions for layer instance
// --------------------------------------------------------------------------

void forward_image_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;
  image_forward(get_top(layer, 0), get_top(layer, 1), net->p_images,
                net->image_heights, net->image_widths, net->num_images,
                (unsigned char*)net->temp_data, &layer->option);
}

void shape_image_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;
  long int temp_space;

  image_shape(get_top(layer, 0), get_top(layer, 1),
              net->image_heights, net->image_widths, net->num_images,
              &temp_space, &layer->option);

  update_temp_space(net, temp_space);
}

void init_image_layer(void* const net_, void* const layer_)
{
  Layer* const layer = (Layer*)layer_;

  // it commonly reduces memory consumption
  get_top(layer, 1)->data_type = PRIVATE_DATA;
}

void free_image_layer(void* const net_, void* const layer_)
{
  return;
}
