#ifndef PVA_DL_NET_H
#define PVA_DL_NET_H

#include "core/layer.h"

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
  //   temp_cpu_data: temp space allocated at main memory
  //   temp_data: in GPU mode, temp space allocated at GPU memory
  //              in CPU mode, temp_data = temp_cpu_data
  real* temp_cpu_data;
  real* temp_data;
  long int temp_space;

  // a constant array [1, 1, ..., 1] used for conv, deconv, fc layers
  real* const_data;
  long int const_space;

  // total size of allocated memory (byte)
  //   space_cpu: size allocated at main memory
  //   space: in GPU mode, size allocated at GPU memory
  //          in CPU mode, space = space_cpu
  long int space_cpu;
  long int space;

  // flag whether this network is ready to run (= 1) or not (= 0)
  int initialized;

  // pointers to input images and their sizes
  unsigned char* p_images[BATCH_SIZE];
  int image_heights[BATCH_SIZE];
  int image_widths[BATCH_SIZE];
  int num_images;

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

Net* create_empty_net(void);

void forward_net(Net* const net);

void shape_net(Net* const net);

void malloc_net(Net* const net);

void free_net(Net* const net);

Tensor* get_tensor_by_name(Net* const net,
                           const char* const name);

Layer* get_layer_by_name(Net* const net,
                         const char* const name);

#ifdef __cplusplus
} // end extern "C"
#endif



// --------------------------------------------------------------------------
// functions not opened to shared library
// --------------------------------------------------------------------------

void update_temp_space(Net* const net, const long int space);

void update_const_space(Net* const net, const long int space);

Tensor* get_tensor(Net* const net,
                   const int tensor_id);

Layer* get_layer(Net* const net,
                 const int layer_id);

int get_tensor_id(Net* const net,
                  const Tensor* const tensor);

int get_layer_id(Net* const net,
                 const Layer* const layer);

Tensor* add_tensor(Net* const net,
                   const char* const name);

Layer* add_layer(Net* const net,
                 const char* const name);

Tensor* find_or_add_tensor(Net* const net,
                           const char* const name);

Layer* find_or_add_layer(Net* const net,
                         const char* const name);

void save_layer_tops(void* const net_,
                     void* const layer_);

void print_layer_tops(void* const net_,
                      void* const layer_);



// --------------------------------------------------------------------------
// simple functions returning static constants
//   required for Python interface
// --------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

int _max_num_tensors(void);

int _max_num_layers(void);

int _max_num_shared_blocks(void);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end PVA_DL_NET_H
