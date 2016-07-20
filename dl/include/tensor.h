#ifndef PVA_DL_TENSOR_H
#define PVA_DL_TENSOR_H

#include "common.h"

// --------------------------------------------------------------------------
// definitions
// --------------------------------------------------------------------------

#ifdef DEMO
  #define BATCH_SIZE 1
#else
  #define BATCH_SIZE 4
#endif

#define MAX_NDIM 5

// data_type values
#define SHARED_DATA 0  // data uses a block of shared memory
#define PRIVATE_DATA 1  // data has its own private memory
#define PARAM_DATA 2  // has own memory & pre-trained parameter is loaded



// --------------------------------------------------------------------------
// data structure
// --------------------------------------------------------------------------

typedef struct Tensor_
{
  char name[MAX_NAME_LEN];
  int num_items;
  int ndim;
  int shape[BATCH_SIZE][MAX_NDIM];
  int start[BATCH_SIZE];
  real* data;
  int data_type;
} Tensor;



// --------------------------------------------------------------------------
// functions
// --------------------------------------------------------------------------

// initialize: set all values to 0
void init_tensor(Tensor* const tensor);

// set tensor's name
void set_tensor_name(Tensor* const tensor, const char* const name);

// total number of elements in a tensor
long int get_data_size(const Tensor* const tensor);

// allocate memory for tensor's data
//   allocate GPU memory in GPU mode, or CPU memory in CPU mode
//   return memory size in bytes
long int malloc_tensor_data(Tensor* const tensor,
                            real* const shared_block);

// deallocate memory
void free_tensor_data(Tensor* const tensor);

// load binary data from file & store to CPU memory
//   cpu_data: pointer to CPU memory for storing data
void load_from_binary_file(const char* const filename,
                           int* const ndim,
                           int shape[],
                           real cpu_data[]);

// load binary data from file & copy to memory where tensor data refers
//   temp_cpu_data: pointer to CPU memory for loading data temporarily
void load_tensor_data(const char* const filename,
                      Tensor* const tensor,
                      real temp_cpu_data[]);

// save tensor data to binary file
//   temp_cpu_data: pointer to CPU memory for storing data temporarily
//                  can be set NULL if tensor data refers to CPU memory
void save_tensor_data(const char* const filename,
                      const Tensor* const tensor,
                      real temp_cpu_data[]);

// print shapes for all batch items in tensor
void print_tensor_info(const Tensor* const tensor);

#endif // end PVA_DL_TENSOR_H
