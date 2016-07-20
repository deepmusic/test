#ifndef PVA_DL_COMMON_H
#define PVA_DL_COMMON_H

// --------------------------------------------------------------------------
// definitions
// --------------------------------------------------------------------------

//#define DEBUG
//#define MKL
#define DEMO

#define MAX_NAME_LEN 64
typedef float real;



// --------------------------------------------------------------------------
// include cuda & blas library
// --------------------------------------------------------------------------

#ifdef GPU
  #include <cublas_v2.h>
  #include <cuda.h>
  #include <cuda_runtime.h>
#elif defined(MKL)
  #include <mkl_cblas.h>
  #include <math.h>
#else
  #include <cblas.h>
  #include <math.h>
#endif

#include <stdio.h>
#include <stdlib.h>



// --------------------------------------------------------------------------
// simple math operators
// --------------------------------------------------------------------------

#define ABS(x)  ((x) > 0 ? (x) : (-(x)))
#define DIV_THEN_CEIL(x, y)  (((x) + (y) - 1) / (y))
#define ROUND(x)  ((int)((x) + 0.5f))

#ifdef GPU
  #define MIN(x, y)  min(x, y)
  #define MAX(x, y)  max(x, y)
#else
  #define MIN(x, y)  ((x) < (y) ? (x) : (y))
  #define MAX(x, y)  ((x) > (y) ? (x) : (y))
#endif

#endif // end PVA_DL_COMMON_H






