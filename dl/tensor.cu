#include "layer.h"
#include <string.h>

// initialize: set all values to 0
void init_tensor(Tensor* const tensor)
{
  memset(tensor, 0, sizeof(Tensor));
}

// allocate memory for tensor
//   allocate GPU memory in GPU mode, or CPU memory in CPU mode
long int malloc_tensor_data(Tensor* const tensor)
{
  if (!tensor->max_data_size) {
    tensor->max_data_size = flatten_size(tensor);
  }

  #ifdef GPU
  cudaMalloc(&tensor->data, tensor->max_data_size * sizeof(real));
  #else
  tensor->data = (real*)malloc(tensor->max_data_size * sizeof(real));
  #endif

  return tensor->max_data_size * sizeof(real);
}

// deallocate memory
long int free_tensor_data(Tensor* const tensor)
{
  #ifdef GPU
  cudaFree(tensor->data);
  #else
  free(tensor->data);
  #endif

  return tensor->max_data_size * sizeof(real);
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
  long int data_size = 1;

  {
    int ndim;
    int shape[MAX_NDIM];

    load_data(filename, &ndim, shape, temp_data);
    for (int i = 0; i < ndim; ++i) {
      data_size *= shape[i];
    }
  }

  {
    if (data_size != flatten_size(tensor)) {
      printf("[ERROR] Size mismatch: %s (%ld) != tensor (%ld)\n",
             filename, data_size, flatten_size(tensor));
      data_size = MIN(data_size,  flatten_size(tensor));
    }

  #ifdef GPU
    cudaMemcpyAsync(tensor->data, temp_data, data_size * sizeof(real),
                    cudaMemcpyHostToDevice);
  #else
    memcpy(tensor->data, temp_data, data_size * sizeof(real));
  #endif
  }
}

// save tensor data to binary file
//   temp_data: pointer to CPU memory for storing data temporarily
//              not used (i.e., can be NULL) if tensor occupies CPU memory
void save_tensor_data(const char* const filename,
                      const Tensor* const tensor,
                      real* const temp_data)
{
  FILE* fp = fopen(filename, "wb");
  real* p_temp_data;

  {
  #ifdef GPU
    p_temp_data = temp_data;
    cudaMemcpyAsync(p_temp_data, tensor->data,
                    flatten_size(tensor) * sizeof(real),
                    cudaMemcpyDeviceToHost);
  #else
    p_temp_data = tensor->data;
  #endif
  }

  for (int n = 0; n < tensor->num_items; ++n)
  {
    int item_size = 1;
    for (int i = 0; i < tensor->ndim; ++i) {
      item_size *= tensor->shape[n][i];
    }

    fwrite(&tensor->ndim, sizeof(int), 1, fp);
    fwrite(tensor->shape[n], sizeof(int), tensor->ndim, fp);
    fwrite(p_temp_data, sizeof(real), item_size, fp);
    p_temp_data += item_size;
  }

  fclose(fp);
}

// total number of elements in a tensor
long int flatten_size(const Tensor* const tensor)
{
  long int total_size = 0;
  for (int n = 0; n < tensor->num_items; ++n) {
    long int size = 1;
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
  #ifdef DEBUG
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
  #endif

  return;
}
