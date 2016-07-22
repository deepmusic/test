#include "layers/operator.h"

// --------------------------------------------------------------------------
// layer-wise operator code
// --------------------------------------------------------------------------

// fully-connected: bottom -> top
//   bottom: N x D (N items of D-dim array)
//   top: N x D' (N items of D-dim array)
//   weight: D' x D
//   bias: 1 x D'
//   const: N-dim array,  const[i] = 1 for all i
static
void fc_forward(const Tensor* const bottom2d,
                Tensor* const top2d,
                const Tensor* const weight2d,
                const Tensor* const bias1d,
                const real const_data[],
                const LayerOption* const option)
{
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
    cblas_sger(CblasRowMajor,
               N,  top_D,
               1,
               const_data,  1,
               bias1d->data,  1,
               top2d->data,  top_D);
  #endif
  }
}



// --------------------------------------------------------------------------
// output & parameter shape calculator code
// --------------------------------------------------------------------------

static
void fc_shape(const Tensor* const bottom2d,
              Tensor* const top2d,
              Tensor* const weight2d,
              Tensor* const bias1d,
              long int* const p_const_space,
              const LayerOption* const option)
{
  // bottom shape: N x D
  const int N = bottom2d->shape[0][0];
  const int bottom_D = bottom2d->shape[0][1]; // D

  // top shape: N x D'
  const int top_D = option->num_output;  // D'
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
  *p_const_space = N * sizeof(real);
}



// --------------------------------------------------------------------------
// functions for layer instance
// --------------------------------------------------------------------------

void forward_fc_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;
  Tensor* const p_bias = (layer->option.bias) ? get_param(layer, 1) : NULL;

  fc_forward(get_bottom(layer, 0), get_top(layer, 0),
             get_param(layer, 0), p_bias,
             net->const_data, &layer->option);
}

void shape_fc_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;
  Tensor* const p_bias = (layer->option.bias) ? get_param(layer, 1) : NULL;
  long int const_space;

  fc_shape(get_bottom(layer, 0), get_top(layer, 0),
           get_param(layer, 0), p_bias,
           &const_space, &layer->option);

  update_const_space(net, const_space);
}

void init_fc_layer(void* const net_, void* const layer_)
{
  return;
}

void free_fc_layer(void* const net_, void* const layer_)
{
  return;
}
