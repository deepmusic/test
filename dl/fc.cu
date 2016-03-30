#include "layer.h"

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
                const real* const const_data,
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
                1.0f,
                bottom2d->data,  bottom_D,
                weight2d->data,  bottom_D,
                0.0f,
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
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,  CblasNoTrans,
                N,  top_D,  1,
                1.0f,
                const_data,  1,
                bias1d->data,  top_D,
                1.0f,
                top2d->data,  top_D);
  #endif
  }
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

  Tensor* p_bias = (layer->option.bias) ? &layer->params[1] : NULL;

  fc_forward(layer->p_bottoms[0], &layer->tops[0],
             &layer->params[0], p_bias,
             net->const_data, &layer->option);
  print_tensor_info(layer->name, &layer->tops[0]);
}

void shape_fc_layer(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;

  int const_size;
  Tensor* p_bias = (layer->option.bias) ? &layer->params[1] : NULL;

  fc_shape(layer->p_bottoms[0], &layer->tops[0],
           &layer->params[0], p_bias,
           &const_size, &layer->option);

  update_net_size(net, layer, 0, 0, const_size);
}



// --------------------------------------------------------------------------
// test code
// --------------------------------------------------------------------------

#ifdef TEST
#include <stdio.h>

int main(int argc, char* argv[])
{
  // variable declaration & memory allocation
  Tensor X, Y, W, b;
  real *X_data = NULL, *Y_data = NULL, *Y_true_data = NULL;
  real *W_data = NULL, *b_data = NULL;
  real *const_data = NULL, *p_const_data = NULL;
  LayerOption option;
  int const_size;

  // set option
  {
    option.out_channels = 84;
    option.bias = 1;
  }

  // load data
  {
    int ndim;
    int shape[g_max_ndim];

    X_data = load_data("../data/temp/fc_bottom0.bin",
                       &ndim, shape, NULL);
    X.num_items = 1;
    X.ndim = ndim;
    for (int i = 0; i < X.ndim; ++i) {
      X.shape[0][i] = shape[i];
    }
    X.start[0] = 0;

    fc_shape(&X, &Y, &W, &b, &const_size, &option);

    Y_true_data = load_data("../data/temp/fc_top0.bin",
                            &ndim, shape, NULL);
    Y_data = (real*)malloc(flatten_size(&Y) * sizeof(real));

    W_data = load_data("../data/temp/fc_param0.bin",
                       &ndim, shape, NULL);

    if (option.bias) {
      b_data = load_data("../data/temp/fc_param1.bin",
                         &ndim, shape, NULL);

      const_data = (real*)malloc(const_size * sizeof(real));
      for (int i = 0; i < const_size; ++i) {
        const_data[i] = 1;
      }
    }
  }
 
  // CUDA initialization
  #ifdef GPU
  {
    printf("set device\n");
    cudaSetDevice(0);
    option.handle = (cublasHandle_t*)malloc(sizeof(cublasHandle_t));
    if (cublasCreate((cublasHandle_t*)option.handle)
          != CUBLAS_STATUS_SUCCESS) {
      printf("cublas creation failed\n");
    }
  }
  #endif

  // bind loaded data to corresponding tensors
  #ifdef GPU
  {
    const long int X_size = flatten_size(&X);
    const long int Y_size = flatten_size(&Y);
    const long int W_size = flatten_size(&W);
    const long int b_size = flatten_size(&b);

    printf("gpu malloc\n");
    cudaMalloc(&X.data, X_size * sizeof(real));
    cudaMalloc(&Y.data, Y_size * sizeof(real));
    cudaMalloc(&W.data, W_size * sizeof(real));
    if (option.bias) {
      cudaMalloc(&b.data, b_size * sizeof(real));
      cudaMalloc(&p_const_data, const_size * sizeof(real));
    }
    else {
      b.data = NULL;
    }

    printf("memcpy: cpu -> gpu\n");
    cudaMemcpyAsync(X.data, X_data, X_size * sizeof(real),
                    cudaMemcpyHostToDevice);
    cudaMemcpyAsync(W.data, W_data, W_size * sizeof(real),
                    cudaMemcpyHostToDevice);
    if (option.bias) {
      cudaMemcpyAsync(b.data, b_data, b_size * sizeof(real),
                      cudaMemcpyHostToDevice);
      cudaMemcpyAsync(p_const_data, const_data, const_size * sizeof(real),
                      cudaMemcpyHostToDevice);
    }
  }
  #else
  {
    X.data = X_data;
    Y.data = Y_data;
    W.data = W_data;
    if (option.bias) {
      b.data = b_data;
      p_const_data = const_data;
    }
    else {
      b.data = NULL;
    }
  }
  #endif

  // do forward operation
  {
    printf("do forward\n");
    fc_forward(&X, &Y, &W, &b, p_const_data, &option);
  }

  // copy GPU data to main memory
  #ifdef GPU
  {
    const long int Y_size = flatten_size(&Y);

    printf("memcpy: cpu <- gpu\n");
    cudaMemcpyAsync(Y_data, Y.data, Y_size * sizeof(real),
                    cudaMemcpyDeviceToHost);
  }
  #endif

  // verify results
  {
    int i = 0;

    printf("verification\n");

    for (int n = 0; n < Y.shape[0][0]; ++n) {
      for (int d = 0; d < Y.shape[0][1]; ++d) {
        real diff = ABS(Y_data[i] - Y_true_data[i]);
        diff /= 1e-10f + MIN(ABS(Y_data[i]), ABS(Y_true_data[i]));
      #ifdef GPU
        if (diff > 0) {
          printf("Y[%d,%d] = %.6f  Y_true[%d,%d] = %.6f\n",
                 n, d, Y_data[i], n, d, Y_true_data[i]);
        }
      #else
        if (diff > 1e-3f) {
          printf("Y[%d,%d] = %.6f  Y_true[%d,%d] = %.6f\n",
                 n, d, Y_data[i], n, d, Y_true_data[i]);
        }
      #endif
        ++i;
      } // endfor d
    } // endfor n
  }

  // memory deallocation
  {
    printf("free\n");
    free(X_data);
    free(Y_data);
    free(Y_true_data);
    free(W_data);
    if (option.bias) {
      free(b_data);
      free(const_data);
    }
  }
  #ifdef GPU
  {
    printf("gpu free\n");
    cudaFree(X.data);
    cudaFree(Y.data);
    cudaFree(W.data);
    if (option.bias) {
      cudaFree(b.data);
      cudaFree(p_const_data);
    }

    if (cublasDestroy(*((cublasHandle_t*)option.handle))
        != CUBLAS_STATUS_SUCCESS) {
      printf("cublas destruction failed\n");
    }
    free(option.handle);
  }
  #endif

  return 0;
}
#endif // endifdef TEST
