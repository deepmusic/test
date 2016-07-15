#include "layer.h"

#include <time.h>

static float a_time[8] = { 0, };
static clock_t tick0, tick1, tick00;

// --------------------------------------------------------------------------
// layer operator code
//   fc_forward
// --------------------------------------------------------------------------

// fully-connected: bottom -> top
//   bottom: N x D (N items of D-dim array)
//   top: N x D' (N items of D-dim array)
//   weight: D' x D
//   bias: 1 x D'
//   const: N-dim array,  const[i] = 1 for all i
void fc_forward(const Tensor* const bottom2d,
                Tensor* const top2d,
                const Tensor* const weight2d,
                const Tensor* const bias1d,
                const real const_data[],
                const LayerOption* const option)
{
  tick00 = clock();

  // weight shape: D' x D
  const int top_D = weight2d->shape[0][0];  // D'
  const int bottom_D = weight2d->shape[0][1]; // D

  // bottom shape: N x D
  const int N = bottom2d->shape[0][0]; // N

  // set top shape: N x D'
  top2d->num_items = 1;
  top2d->ndim = 2;
  top2d->shape[0][0] = N;
  top2d->shape[0][1] = top_D;

  tick0 = clock();
  // compute top = dot(bottom, weight.transpose())
  //   bottom: N x D
  //   weight: D' x D
  //   top: N x D'
  {
    // compute Z = alpha * dot(X, Y) + beta * Z
    //   X (= bottom): m x p,  Y (= weight): n x p,  Z (= top): m x n
    //   X, Y, Z: row-major order (e.g., Z[i][j] = Z[i * n + j])
  #ifdef GPU
    // input arguments:
    //   cublas handle,
    //   do_transpose_Y (= true),  do_transpose_X (= false),
    //   n (= D'),  m (= N),  p (= D),
    //   &alpha (= 1),
    //   &Y,  number of columns in Y (= D),
    //   &X,  number of columns in X (= D),
    //   &beta (= 0),
    //   &Z,  number of columns in Z (= D')
    const real one = 1.0f, zero = 0.0f;
    cublasSgemm(*((cublasHandle_t*)option->handle),
                CUBLAS_OP_T,  CUBLAS_OP_N,
                top_D,  N,  bottom_D,
                &one,
                weight2d->data,  bottom_D,
                bottom2d->data,  bottom_D,
                &zero,
                top2d->data,  top_D);
  #else
    // input arguments:
    //   is_row_major_order (= true),
    //   do_transpose_X (= false),  do_transpose_Y (= true),
    //   m (= N),  n (= D'),  p (= D),
    //   alpha (= 1),
    //   &X,  number of columns in X (= D),
    //   &Y,  number of columns in Y (= D),
    //   beta (= 0),
    //   &Z,  number of columns in Z (= D')
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,  CblasTrans,
                N,  top_D,  bottom_D,
                1,
                bottom2d->data,  bottom_D,
                weight2d->data,  bottom_D,
                0,
                top2d->data,  top_D);
  #endif
  }
  tick1 = clock();
  a_time[0] = (float)(tick1 - tick0) / CLOCKS_PER_SEC;

  tick0 = clock();
  // compute top[i][j] = top[i][j] + bias[j]
  //   top: N x D'
  //   bias: 1 x D'
  if (option->bias) {
    // the computation is equivalent to...
    //   top = top + dot(constant, bias)
    //   constant: N x 1,  constant[i] = 1 for all i
  #ifdef GPU
    // thus, input arguments:
    //   do_transpose_Y (= false),  do_transpose_X (= false),
    //   n = D',  m = N,  p = 1
    //   alpha = 1,  beta = 1
    const real one = 1.0f;
    cublasSgemm(*((cublasHandle_t*)option->handle),
                CUBLAS_OP_N,  CUBLAS_OP_N,
                top_D,  N,  1,
                &one,
                bias1d->data,  top_D,
                const_data,  1,
                &one,
                top2d->data,  top_D);
  #else
    // input arguments:
    //   do_transpose_X (= false),  do_transpose_Y (= false),
    //   m = N,  n = D',  p = 1
    //   alpha = 1,  beta = 1
/*
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,  CblasNoTrans,
                N,  top_D,  1,
                1,
                const_data,  1,
                bias1d->data,  top_D,
                1,
                top2d->data,  top_D);
*/
    cblas_sger(CblasRowMajor,
               N,  top_D,
               1,
               const_data,  1,
               bias1d->data,  1,
               top2d->data,  top_D);
  #endif
  }
  tick1 = clock();
  a_time[1] = (float)(tick1 - tick0) / CLOCKS_PER_SEC;
  a_time[6] = (float)(tick1 - tick00) / CLOCKS_PER_SEC;
  a_time[7] += (float)(tick1 - tick00) / CLOCKS_PER_SEC;
}



// --------------------------------------------------------------------------
// layer shape calculator code
// --------------------------------------------------------------------------

void fc_shape(const Tensor* const bottom2d,
              Tensor* const top2d,
              Tensor* const weight2d,
              Tensor* const bias1d,
              int* const const_size,
              const LayerOption* const option)
{
  // bottom shape: N x D
  const int N = bottom2d->shape[0][0];
  const int bottom_D = bottom2d->shape[0][1]; // D

  // top shape: N x D'
  const int top_D = option->out_channels;  // D'
  top2d->num_items = 1;
  top2d->ndim = 2;
  top2d->shape[0][0] = N;
  top2d->shape[0][1] = top_D;
  top2d->start[0] = 0;

  // weight shape: D' x D
  weight2d->num_items = 1;
  weight2d->ndim = 2;
  weight2d->shape[0][0] = top_D;
  weight2d->shape[0][1] = bottom_D;
  weight2d->start[0] = 0;

  // bias shape: D' x 1
  if (option->bias) {
    bias1d->num_items = 1;
    bias1d->ndim = 1;
    bias1d->shape[0][0] = top_D;
    bias1d->start[0] = 0;
  }
  else if (bias1d) {
    bias1d->num_items = 0;
    bias1d->ndim = 0;
    bias1d->shape[0][0] = 0;
    bias1d->start[0] = 0;
  }

  // constant data size: N
  *const_size = N;
}



// --------------------------------------------------------------------------
// API code
// --------------------------------------------------------------------------

void forward_fc_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;
  Tensor* const p_bias = (layer->option.bias) ? layer->p_params[1] : NULL;

  fc_forward(layer->p_bottoms[0], layer->p_tops[0],
             layer->p_params[0], p_bias,
             net->const_data, &layer->option);

  #ifdef DEBUG
  {
    printf("%s:  ", layer->name);
    for (int i = 0; i < 8; ++i) {
      printf("%4.2f\t", a_time[i] * 1000);
    }
    printf("\n");
  }
  #endif
}

void shape_fc_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;
  Tensor* const p_bias = (layer->option.bias) ? layer->p_params[1] : NULL;
  int const_size;

  fc_shape(layer->p_bottoms[0], layer->p_tops[0],
           layer->p_params[0], p_bias,
           &const_size, &layer->option);

  update_net_size(net, layer, 0, 0, const_size);
}
