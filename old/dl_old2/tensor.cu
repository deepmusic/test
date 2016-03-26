#include "layer.h"
#include <stdio.h>

// total number of elements in a tensor
int flatten_size(const Tensor* const tensor)
{
  int total_size = 0;
  for (int n = 0; n < tensor->num_items; ++n) {
    int size = 1;
    for (int d = 0; d < tensor->ndim; ++d) {
      size *= tensor->shape[n][d];
    }
    total_size += size;
  }
  return total_size;
}

// print shapes for all batch items in tensor
void print_tensor_info(const char* const name,
                       const Tensor* const tensor)
{
  printf("%s: ", name);
  if (tensor->num_items > 1) {
    printf("batch size = %d\n", tensor->num_items);
    for (int n = 0; n < tensor->num_items; ++n) {
      printf("  ");
      for (int i = 0; i < tensor->ndim - 1; ++i) {
        printf("%d x ", tensor->shape[n][i]);
      }
      printf("%d, ", tensor->shape[n][tensor->ndim - 1]);
      printf("start = %d\n", tensor->start[n]);
    }
  }
  else {
    for (int i = 0; i < tensor->ndim - 1; ++i) {
      printf("%d x ", tensor->shape[0][i]);
    }
    printf("%d\n", tensor->shape[0][tensor->ndim - 1]);
  }
}

// allocate memory for tensor
//   allocate GPU memory in GPU mode, or CPU memory in CPU mode
int malloc_tensor(Tensor* const tensor)
{
  const int data_size = flatten_size(tensor);

  #ifdef GPU
  cudaMalloc(&tensor->data, data_size * sizeof(real));
  #else
  tensor->data = (real*)malloc(data_size * sizeof(real));
  #endif

  return data_size * sizeof(real);
}

// load binary data from file & store to CPU memory
//   data: pointer to CPU memory for storing data
//         if NULL, allocate new memory & load data & return pointer
real* load_data(const char* const filename,
                int* const ndim,
                int* const shape,
                real* data)
{
  FILE* fp = fopen(filename, "rb");

  // load data shape
  {
    if ((int)fread(ndim, sizeof(int), 1, fp) < 1) {
      printf("Error while reading ndim from %s\n", filename);
    }
    if ((int)fread(shape, sizeof(int), *ndim, fp) != *ndim) {
      printf("Error while reading shape from %s\n", filename);
    }
  }

  // compute total number of elements
  {
    const int ndim_ = *ndim;
    int count = 1;
    for (int i = 0; i < ndim_; ++i) {
      count *= shape[i];
    }
    shape[ndim_] = count;
  }

  // memory allocation & load data
  {
    const int count = shape[*ndim];
    if (data == NULL) {
      data = (real*)malloc(count * sizeof(real));
    }
    if ((int)fread(data, sizeof(real), count, fp) != count) {
      printf("Error while reading data from %s\n", filename);
    }

    // file close & return data
    fclose(fp);
    return data;
  }
}

// load binary data from file & copy to memory where tensor occupies
//   temp_data: pointer to CPU memory for loading data temporarily
//              not used (i.e., can be NULL) if tensor occupies CPU memory
void load_tensor(const char* const filename,
                 Tensor* const tensor,
                 real* const temp_data)
{
  int ndim;
  int shape[g_max_ndim];

  {
  #ifdef GPU
    int data_size = 1;
    load_data(filename, &ndim, shape, temp_data);
    for (int i = 0; i < ndim; ++i) {
      data_size *= shape[i];
    }
    if (data_size != flatten_size(tensor)) {
      printf("[ERROR] Size mismatch: %s (%d) != tensor (%d)\n",
             filename, data_size, flatten_size(tensor));
    }
    cudaMemcpyAsync(tensor->data, temp_data, data_size * sizeof(real),
                    cudaMemcpyHostToDevice);
  #else
    load_data(filename, &ndim, shape, tensor->data);
  #endif
  }
}
