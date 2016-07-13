#include "layer.h"
#include <string.h>

// initialize: set all values to 0
void init_tensor(Tensor* const tensor)
{
  memset(tensor, 0, sizeof(Tensor));
}

void set_tensor_name(Tensor* const tensor, const char* const name)
{
  strcpy(tensor->name, name);
}

// allocate memory for tensor
//   allocate GPU memory in GPU mode, or CPU memory in CPU mode
long int malloc_tensor_data(Tensor* const tensor,
                            real* const shared_blocks[])
{
  long int space = 0;

  if (tensor->max_data_size) {
    printf("[ERROR] Data size %ld for tensor %s seems not be initialized\n",
           tensor->max_data_size, tensor->name);
    return space;
  }
  if (tensor->data) {
    printf("[ERROR] Tensor %s already refers to some memory\n",
           tensor->name);
    return space;
  }

  tensor->max_data_size = flatten_size(tensor);

  if (tensor->data_type != SHARED_DATA) {
    #ifdef GPU
    cudaMalloc(&tensor->data, tensor->max_data_size * sizeof(real));
    #else
    tensor->data = (real*)malloc(tensor->max_data_size * sizeof(real));
    #endif
    space = tensor->max_data_size * sizeof(real);

    #ifdef DEBUG
    printf("%s: Memory allocated, %ld byte\n", tensor->name, space);
    #endif
  }
  else {
    tensor->data = shared_blocks[tensor->shared_block_id];
    #ifdef DEBUG
    printf("%s: Shared memory assigned, block %d\n",
           tensor->name, tensor->shared_block_id);
    #endif
  }

  return space;
}

// deallocate memory
long int free_tensor_data(Tensor* const tensor)
{
  long int space = 0;

  if (tensor->data_type != SHARED_DATA) {
    if (tensor->data) {
      #ifdef GPU
      cudaFree(tensor->data);
      #else
      free(tensor->data);
      #endif
      tensor->data = NULL;
      space = tensor->max_data_size * sizeof(real);
    }
    else {
      printf("[ERROR] Tensor %s: Data memory was not allocated\n",
             tensor->name);
    }
  }

  return space;
}

// load binary data from file & store to CPU memory
//   data: pointer to CPU memory for storing data
void load_from_binary_file(const char* const filename,
                           int* const ndim,
                           int shape[],
                           real cpu_data[])
{
  FILE* fp = fopen(filename, "rb");

  if (!fp) {
    printf("[ERROR] File not found: %s\n", filename);
    return;
  }

  // load data shape
  {
    if ((int)fread(ndim, sizeof(int), 1, fp) < 1) {
      printf("[ERROR] I/O error while reading ndim from %s\n", filename);
    }
    if ((int)fread(shape, sizeof(int), *ndim, fp) != *ndim) {
      printf("[ERROR] I/O error while reading shape from %s\n", filename);
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
    if (!cpu_data) {
      printf("[ERROR] NULL pointer to memory for data loading\n");
    }
    else if ((int)fread(cpu_data, sizeof(real), count, fp) != count) {
      printf("[ERROR] I/O error while reading data from %s\n", filename);
    }

  }

  // file close
  fclose(fp);
}

// load binary data from file & copy to memory where tensor occupies
//   temp_cpu_data: pointer to CPU memory for loading data temporarily
void load_tensor_data(const char* const filename,
                      Tensor* const tensor,
                      real temp_cpu_data[])
{
  long int data_size = 1;

  if (temp_cpu_data) {
    int ndim;
    int shape[MAX_NDIM];

    load_from_binary_file(filename, &ndim, shape, temp_cpu_data);
    for (int i = 0; i < ndim; ++i) {
      data_size *= shape[i];
    }
  }
  else {
    printf("[ERROR] NULL pointer to temp memory for data loading\n");
    return;
  }

  if (data_size != flatten_size(tensor)) {
    printf("[ERROR] Size mismatch: %s (%ld) != tensor (%ld)\n",
           filename, data_size, flatten_size(tensor));
    data_size = MIN(data_size,  flatten_size(tensor));
  }

  #ifdef GPU
  cudaMemcpyAsync(tensor->data, temp_cpu_data, data_size * sizeof(real),
                  cudaMemcpyHostToDevice);
  #else
  memcpy(tensor->data, temp_cpu_data, data_size * sizeof(real));
  #endif
}

// save tensor data to binary file
//   temp_cpu_data: pointer to CPU memory for storing data temporarily
void save_tensor_data(const char* const filename,
                      const Tensor* const tensor,
                      real temp_cpu_data[])
{
  FILE* fp = fopen(filename, "wb");
  real* p_temp_cpu_data;

  {
  #ifdef GPU
    p_temp_cpu_data = temp_cpu_data;
    cudaMemcpyAsync(p_temp_cpu_data, tensor->data,
                    flatten_size(tensor) * sizeof(real),
                    cudaMemcpyDeviceToHost);
  #else
    p_temp_cpu_data = tensor->data;
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
    fwrite(p_temp_cpu_data, sizeof(real), item_size, fp);
    p_temp_cpu_data += item_size;
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
    printf("%s (size = %ld): ", name, flatten_size(tensor));
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
