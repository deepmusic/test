#include "core/tensor.h"
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

// total number of elements in a tensor
long int get_data_size(const Tensor* const tensor)
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

// allocate memory for tensor
//   allocate GPU memory in GPU mode, or CPU memory in CPU mode
long int malloc_tensor_data(Tensor* const tensor,
                            real* const shared_block)
{
  long int space = get_data_size(tensor) * sizeof(real);

  if (tensor->data) {
    printf("[ERROR] Tensor %s already refers to some memory\n",
           tensor->name);
    return space;
  }

  if (tensor->data_type != SHARED_DATA) {
    #ifdef GPU
    cudaMalloc(&tensor->data, space);
    cudaMemset(tensor->data, 0, space);
    #else
    tensor->data = (real*)malloc(space);
    memset(tensor->data, 0, space);
    #endif
  }
  else {
    tensor->data = shared_block;
    space = 0;
  }

  #ifdef DEBUG
  {
    if (tensor->data_type != SHARED_DATA) {
      printf("%s: Memory allocated, %ld byte\n", tensor->name, space);
    }
  }
  #endif

  return space;
}

// deallocate memory
void free_tensor_data(Tensor* const tensor)
{
  if (!tensor->data) {
    printf("[ERROR] Tensor %s: Data memory was not allocated\n",
           tensor->name);
  }

  else if (tensor->data_type != SHARED_DATA) {
    #ifdef GPU
    cudaFree(tensor->data);
    #else
    free(tensor->data);
    #endif
    tensor->data = NULL;
  }
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

  if (data_size != get_data_size(tensor)) {
    printf("[ERROR] Size mismatch: %s (%ld) != tensor (%ld)\n",
           filename, data_size, get_data_size(tensor));
    data_size = MIN(data_size,  get_data_size(tensor));
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
                    get_data_size(tensor) * sizeof(real),
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

// print shapes for all batch items in tensor
void print_tensor_info(const Tensor* const tensor)
{
  #ifdef DEBUG
  {
    printf("%s (size = %ld): ", tensor->name, get_data_size(tensor));
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
}
