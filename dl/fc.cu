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
                const FCOption* const option)
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
// test code
// --------------------------------------------------------------------------

#ifdef TEST
#include <stdio.h>
#include <stdlib.h>

#define IN_DATA_SIZE 300*4096
#define OUT_DATA_SIZE 300*84
#define WEIGHT_SIZE 84*4096
#define BIAS_SIZE 84
#define CONST_SIZE 300

int main(int argc, char *argv[])
{
  // variable declaration & memory allocation
  Tensor X, Y, W, b;
  real* const X_data = (real*)malloc(IN_DATA_SIZE * sizeof(real));
  real* const Y_data = (real*)malloc(OUT_DATA_SIZE * sizeof(real));
  real* const Y_true_data = (real*)malloc(OUT_DATA_SIZE * sizeof(real));
  real* const W_data = (real*)malloc(WEIGHT_SIZE * sizeof(real));
  real* const b_data = (real*)malloc(BIAS_SIZE * sizeof(real));
  real* const const_data = (real*)malloc(CONST_SIZE * sizeof(real));
  real* p_const_data;
  FCOption option;

  // set option
  {
    option.out_channels = 84;
    option.bias = 1;
  }

  // set data shapes
  {
    X.ndim = 2; X.num_items = 1;
    X.shape[0][0] = 300;
    X.shape[0][1] = 4096;

    Y.ndim = X.ndim; Y.num_items = 1;
    Y.shape[0][0] = X.shape[0][0];
    Y.shape[0][1] = option.out_channels;

    W.ndim = 2; W.num_items = 1;
    W.shape[0][0] = option.out_channels;
    W.shape[0][1] = X.shape[0][1];

    b.ndim = 1; b.num_items = 1;
    b.shape[0][0] = option.out_channels;
  }
 
  // load data
  {
    FILE* fp;
    const int X_size = flatten_size(&X);
    const int Y_size = flatten_size(&Y);
    const int W_size = flatten_size(&W);
    const int b_size = flatten_size(&b);

    printf("data loading\n");

    fp = fopen("../data/temp/fc_bottom0.bin", "rb");
    if ((int)fread(X_data, sizeof(real), X_size, fp) != X_size) {
      printf("Error while reading fc_bottom0\n");
    }
    fclose(fp);

    fp = fopen("../data/temp/fc_param0.bin", "rb");
    if ((int)fread(W_data, sizeof(real), W_size, fp) != W_size) {
      printf("Error while reading fc_param0\n");
    }
    fclose(fp);

    if (option.bias) {
      fp = fopen("../data/temp/fc_param1.bin", "rb");
      if ((int)fread(b_data, sizeof(real), b_size, fp) != b_size) {
        printf("Error while reading fc_param1\n");
      }
      fclose(fp);

      for (int i = 0; i < CONST_SIZE; ++i) {
        const_data[i] = 1;
      }
    }

    fp = fopen("../data/temp/fc_top0.bin", "rb");
    if ((int)fread(Y_true_data, sizeof(real), Y_size, fp) != Y_size) {
      printf("Error while reading fc_top0\n");
    }
    fclose(fp);
  }

  // CUDA initialization
  #ifdef GPU
  {
    printf("set device\n");
    CUDA_CHECK(cudaSetDevice(0));
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
    const int X_size = flatten_size(&X);
    const int Y_size = flatten_size(&Y);
    const int W_size = flatten_size(&W);
    const int b_size = flatten_size(&b);

    printf("gpu malloc\n");
    CUDA_CHECK(cudaMalloc(&X.data, X_size * sizeof(real)));
    CUDA_CHECK(cudaMalloc(&Y.data, Y_size * sizeof(real)));
    CUDA_CHECK(cudaMalloc(&W.data, W_size * sizeof(real)));
    CUDA_CHECK(cudaMalloc(&b.data, b_size * sizeof(real)));
    CUDA_CHECK(cudaMalloc(&p_const_data, CONST_SIZE * sizeof(real)));

    printf("memcpy: cpu -> gpu\n");
    CUDA_CHECK(cudaMemcpy(X.data, X_data, X_size * sizeof(real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(W.data, W_data, W_size * sizeof(real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.data, b_data, b_size * sizeof(real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p_const_data, const_data,
                          CONST_SIZE * sizeof(real),
                          cudaMemcpyHostToDevice));
  }
  #else
  {
    X.data = X_data;
    Y.data = Y_data;
    W.data = W_data;
    b.data = b_data;
    p_const_data = const_data;
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
    const int Y_size = flatten_size(&Y);

    printf("memcpy: cpu <- gpu\n");
    CUDA_CHECK(cudaMemcpy(Y_data, Y.data, Y_size * sizeof(real),
                          cudaMemcpyDeviceToHost));
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
    free(b_data);
    free(const_data);
  }
  #ifdef GPU
  {
    printf("gpu free\n");
    CUDA_CHECK(cudaFree(X.data));
    CUDA_CHECK(cudaFree(Y.data));
    CUDA_CHECK(cudaFree(W.data));
    CUDA_CHECK(cudaFree(b.data));
    CUDA_CHECK(cudaFree(p_const_data));

    if (cublasDestroy(*((cublasHandle_t*)option.handle)) != CUBLAS_STATUS_SUCCESS) {
      printf("cublas destruction failed\n");
    }
    free(option.handle);
  }
  #endif

  return 0;
}
#endif // endifdef TEST
