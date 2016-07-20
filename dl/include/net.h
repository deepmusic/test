#ifndef PVA_DL_NET_H
#define PVA_DL_NET_H

#include "tensor.h"
#include "layer.h"

// --------------------------------------------------------------------------
// definitions
// --------------------------------------------------------------------------

#define MAX_NUM_TENSORS 400
#define MAX_NUM_LAYERS 400
#define MAX_NUM_SHARED_BLOCKS 10



// --------------------------------------------------------------------------
// data structure
// --------------------------------------------------------------------------

typedef struct Net_
{
  // path in which pre-trained parameters are located
  char param_path[1024];

  // all tensor instances in this network instance
  Tensor tensors[MAX_NUM_TENSORS];
  int num_tensors;

  // all layer instances in this network instance
  Layer layers[MAX_NUM_LAYERS];
  int num_layers;

  // shared memory blocks for tensors
  //   designed for efficient use of memory
  //   a tensor of "SHARED_DATA" type (by default) uses one of these shared
  //   memory blocks to load or store its data
  real* p_shared_blocks[MAX_NUM_SHARED_BLOCKS];
  int num_shared_blocks;

  // temporary space for some layer-wise operators
  //   temp_data: space allocated at GPU memory (in GPU mode) or main memory
  //   temp_cpu_data: space allocated at main memory (even in GPU mode)
  real* temp_data;
  real* temp_cpu_data;
  long int temp_space;

  // a constant array [1, 1, ..., 1] used for conv, deconv, fc layers
  real* const_data;
  long int const_space;

  // total size of allocated memory (byte)
  //   space_cpu: main memory size (even in GPU mode)
  //   space: GPU memory size (in GPU mode) or main memory size
  //   in CPU mode, total size = space_cpu + space
  long int space_cpu;
  long int space;

  // flag whether this network is ready to run (= 1) or not (= 0)
  int initialized;

  int input_scale;
  int num_output_boxes;

  // auxiliary data for CuBLAS library
  #ifdef GPU
  cublasHandle_t blas_handle;
  #else
  int blas_handle;
  #endif
} Net;



// --------------------------------------------------------------------------
// functions
//   callable from external environments via shared library
// --------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

void init_net(Net* const net);

void malloc_net(Net* const net);

void free_net(Net* const net);

void forward_net(Net* const net);

void shape_net(Net* const net);

Tensor* get_tensor(Net* const net,
                   const int tensor_id);

Layer* get_layer(Net* const net,
                 const int layer_id);

int get_tensor_id(Net* const net,
                  const Tensor* const tensor);

int get_layer_id(Net* const net,
                 const Layer* const layer);

Tensor* find_tensor_by_name(Net* const net,
                            const char* const name);

Layer* find_layer_by_name(Net* const net,
                          const char* const name);

Tensor* add_tensor(Net* const net,
                   const char* const name);

Layer* add_layer(Net* const net,
                 const char* const name);

Tensor* find_or_add_tensor(Net* const net,
                           const char* const name);

Layer* find_or_add_layer(Net* const net,
                         const char* const name);

Tensor* get_tensor_by_name(Net* const net,
                           const char* const name);

Layer* get_layer_by_name(Net* const net,
                         const char* const name);

void save_layer_tops(void* const net_,
                     void* const layer_);

void print_layer_tops(void* const net_,
                      void* const layer_);

#ifdef __cplusplus
} // end extern "C"
#endif



// --------------------------------------------------------------------------
// functions not opened to shared library
// --------------------------------------------------------------------------

void update_temp_space(Net* const net, const long int space);

void update_const_space(Net* const net, const long int space);

void init_input_layer(Net* const net,
                      Tensor* const input3d,
                      Tensor* const img_info1d);

#endif // end PVA_DL_NET_H
