#include "core/net.h"
#include <string.h>

Net* create_empty_net(void)
{
  Net* const net = (Net*)malloc(sizeof(Net));
  
  memset(net, 0, sizeof(Net));

  return net;
}

void forward_net(Net* const net)
{
  if (!net->initialized) {
    printf("[ERROR] Network is not initialized\n");
    return;
  }

  for (int i = 0; i < net->num_layers; ++i) {
    Layer* const layer = get_layer(net, i);
    if (layer->f_forward) {
      (*layer->f_forward)(net, layer);
    }
  }
}

void shape_net(Net* const net)
{
  for (int i = 0; i < net->num_layers; ++i) {
    Layer* const layer = get_layer(net, i);
    if (layer->f_shape) {
      (*layer->f_shape)(net, layer);
    }

    #ifdef DEBUG
    printf("[Layer %s] ", layer->name);
    if (layer->f_shape) {
      for (int top_id = 0; top_id < layer->num_tops; ++top_id) {
        const Tensor* const tensor = get_top(layer, top_id);
        print_tensor_info(tensor);
      }
    }
    else {
      printf("\n");
    }
    #endif
  }
}

Tensor* get_tensor(Net* const net, const int tensor_id)
{
  #ifdef DEBUG
  if (tensor_id >= net->num_tensors) {
    printf("[ERROR] Net: out-of-bound tensor index %d\n", tensor_id);
    return NULL;
  }
  #endif
  return &net->tensors[tensor_id];
} 

Layer* get_layer(Net* const net, const int layer_id)
{
  #ifdef DEBUG
  if (layer_id >= net->num_layers) {
    printf("[ERROR] Net: out-of-bound layer index %d\n", layer_id);
    return NULL;
  }
  #endif
  return &net->layers[layer_id];
} 

int get_tensor_id(Net* const net, const Tensor* const tensor)
{
  const int tensor_id = tensor - net->tensors;

  #ifdef DEBUG
  if (tensor_id < 0 || tensor_id >= net->num_tensors) {
    printf("[ERROR] Tensor %s: located at out-of-bound index %d\n",
           tensor->name, tensor_id);
  }
  else if (tensor != get_tensor(net, tensor_id)) {
    printf("[ERROR] Tensor %s: collision with tensor %s\n",
           tensor->name, get_tensor(net, tensor_id)->name);
  }
  #endif

  return tensor_id;
}

int get_layer_id(Net* const net, const Layer* const layer)
{
  const int layer_id = layer - net->layers;

  #ifdef DEBUG
  if (layer_id < 0 || layer_id >= net->num_layers) {
    printf("[ERROR] Layer %s: located at out-of-bound index %d\n",
           layer->name, layer_id);
  }
  else if (layer != get_layer(net, layer_id)) {
    printf("[ERROR] Layer %s: collision with layer %s\n",
           layer->name, get_layer(net, layer_id)->name);
  }
  #endif

  return layer_id;
}

static
Tensor* find_tensor_by_name(Net* const net, const char* const name)
{
  for (int i = 0; i < net->num_tensors; ++i) {
    Tensor* const tensor = get_tensor(net, i);
    if (strcmp(tensor->name, name) == 0) {
      return tensor;
    }
  }
  return NULL;
}

static
Layer* find_layer_by_name(Net* const net, const char* const name)
{
  for (int i = 0; i < net->num_layers; ++i) {
    Layer* const layer = get_layer(net, i);
    if (strcmp(layer->name, name) == 0) {
      return layer;
    }
  }
  return NULL;
}

Tensor* add_tensor(Net* const net, const char* const name)
{
  {
    Tensor* const tensor = find_tensor_by_name(net, name);
    if (tensor) {
      printf("[ERROR] Net: Tensor %s already exists\n", name);
      return tensor;
    }
  }

  if (net->num_tensors == MAX_NUM_TENSORS) {
    printf("[ERROR] Net: cannot add more tensor\n");
    return NULL;
  }
  ++net->num_tensors;

  {
    Tensor* const tensor = get_tensor(net, net->num_tensors - 1);
    set_tensor_name(tensor, name);
    return tensor;
  }
}

Layer* add_layer(Net* const net, const char* const name)
{
  {
    Layer* const layer = find_layer_by_name(net, name);
    if (layer) {
      printf("[ERROR] Net: Layer %s already exists\n", name);
      return layer;
    }
  }

  if (net->num_layers == MAX_NUM_LAYERS) {
    printf("[ERROR] Net: cannot add more layer\n");
    return NULL;
  }
  ++net->num_layers;

  {
    Layer* const layer = get_layer(net, net->num_layers - 1);
    strcpy(layer->name, name);
    return layer;
  }
}

Tensor* find_or_add_tensor(Net* const net, const char* const name)
{
  Tensor* tensor = find_tensor_by_name(net, name);
  if (!tensor) {
    tensor = add_tensor(net, name);
  }
  return tensor;
}

Layer* find_or_add_layer(Net* const net, const char* const name)
{
  Layer* layer = find_layer_by_name(net, name);
  if (!layer) {
    layer = add_layer(net, name);
  }
  return layer;
}

Tensor* get_tensor_by_name(Net* const net, const char* const name)
{
  Tensor* const tensor = find_tensor_by_name(net, name);
  if (!tensor) {
    printf("[ERROR] Cannot find tensor %s\n", name);
  }
  return tensor;
}

Layer* get_layer_by_name(Net* const net, const char* const name)
{
  Layer* const layer = find_layer_by_name(net, name);
  if (!layer) {
    printf("[ERROR] Cannot find layer %s\n", name);
  }
  return layer;
}

void update_temp_space(Net* const net, const long int space)
{
  if (!net->initialized) {
    net->temp_space = MAX(net->temp_space,  space);
  }
}

void update_const_space(Net* const net, const long int space)
{
  if (!net->initialized) {
    net->const_space = MAX(net->const_space,  space);
  }
}

static
void init_net(Net* const net)
{
  if (net->initialized) {
    printf("[ERROR] Network is already initialized\n");
    return;
  }

  for (int i = 0; i < net->num_layers; ++i) {
    Layer* const layer = get_layer(net, i);
    if (layer->f_init) {
      (*layer->f_init)(net, layer);
    }
  }
}

static
void assign_shared_blocks(Net* const net, int shared_block_id[])
{
  int is_assigned[MAX_NUM_TENSORS] = { 0, };
  int alive_until[MAX_NUM_TENSORS] = { 0, };
  int reserved_until[MAX_NUM_SHARED_BLOCKS] = { 0, };

  // compute lifetime for each tensor
  for (int layer_id = net->num_layers - 1; layer_id >= 0; --layer_id) {
    Layer* const layer = get_layer(net, layer_id);
    for (int bottom_id = 0; bottom_id < layer->num_bottoms; ++bottom_id) {
      Tensor* const tensor = get_bottom(layer, bottom_id);
      const int tensor_id = get_tensor_id(net, tensor);

      if (!alive_until[tensor_id]) {
        alive_until[tensor_id] = layer_id;
      }
    }
  }

  // lifetime for output tensors
  for (int layer_id = 0; layer_id < net->num_layers; ++layer_id) {
    Layer* const layer = get_layer(net, layer_id);
    for (int top_id = 0; top_id < layer->num_tops; ++top_id) {
      Tensor* const tensor = get_top(layer, top_id);
      const int tensor_id = get_tensor_id(net, tensor);

      if (!alive_until[tensor_id]) {
        alive_until[tensor_id] = net->num_layers - 1;
      }
    }
  }

  // assign shared blocks to each tensor according to its lifetime
  for (int layer_id = 0; layer_id < net->num_layers; ++layer_id) {
    Layer* const layer = get_layer(net, layer_id);

    for (int top_id = 0; top_id < layer->num_tops; ++top_id) {
      Tensor* const tensor = get_top(layer, top_id);
      const int tensor_id = get_tensor_id(net, tensor);

      if (tensor->data_type == SHARED_DATA && !is_assigned[tensor_id]) {
        for (int blk_id = 0; blk_id < net->num_shared_blocks; ++blk_id) {
          if (!reserved_until[blk_id]) {
            shared_block_id[tensor_id] = blk_id;
            reserved_until[blk_id] = alive_until[tensor_id];
            is_assigned[tensor_id] = 1;
            break;
          }
        }

        if (!is_assigned[tensor_id]) {
          if (net->num_shared_blocks == MAX_NUM_SHARED_BLOCKS) {
            printf("[ERROR] Failed to assign shared block for %s\n",
                   tensor->name);
          }
          else {
            const int blk_id = net->num_shared_blocks++;
            shared_block_id[tensor_id] = blk_id;
            reserved_until[blk_id] = alive_until[tensor_id];
            is_assigned[tensor_id] = 1;
          }          
        }
      }

      for (int blk_id = 0; blk_id < net->num_shared_blocks; ++blk_id) {
        if (reserved_until[blk_id] == layer_id) {
          reserved_until[blk_id] = 0;
        }
      }
    }
  }

  #ifdef DEBUG
  for (int tensor_id = 0; tensor_id < net->num_tensors; ++tensor_id) {
    if (is_assigned[tensor_id]) {
      const Tensor* const tensor = get_tensor(net, tensor_id);
      printf("Tensor %s: Assigned shared memory block %d, ",
             tensor->name, shared_block_id[tensor_id]);
      printf("reserved until layer %s\n",
             get_layer(net, alive_until[tensor_id])->name);
    }
  }
  #endif
}

void malloc_net(Net* const net)
{
  long int shared_data_space, param_data_space, tensor_space;
  long int layer_space, layer_cpu_space, temp_cpu_space;
  int shared_block_id[MAX_NUM_TENSORS] = { 0, };

  if (net->initialized) {
    printf("[ERROR] Network is already initialized\n");
    return;
  }

  // initialize all layers
  init_net(net);

  // network's space at this time = space for all layers
  #ifdef GPU
  {
    layer_cpu_space = net->space_cpu;
    layer_space = net->space;
  }
  #else
  {
    layer_cpu_space = net->space + net->space_cpu;
    layer_space = layer_cpu_space;
  }
  #endif

  // calculate output shapes and parameter shapes for all layers
  shape_net(net);

  // shared block assignment for all tensors
  assign_shared_blocks(net, shared_block_id);

  // compute maximum size of shared data tensors and parameter tensors
  shared_data_space = 0, param_data_space = 0;
  for (int i = 0; i < net->num_tensors; ++i) {
    const Tensor* const tensor = get_tensor(net, i);
    if (tensor->data_type == SHARED_DATA) {
      shared_data_space = MAX(shared_data_space,  get_data_size(tensor));
    }
    else if (tensor->data_type == PARAM_DATA) {
      param_data_space = MAX(param_data_space,  get_data_size(tensor));
    }
  }
  shared_data_space *= sizeof(real);
  param_data_space *= sizeof(real);

  // memory allocation for shared memory blocks
  for (int i = 0; i < net->num_shared_blocks; ++i) {
    #ifdef GPU
    cudaMalloc(&net->p_shared_blocks[i], shared_data_space);
    cudaMemset(net->p_shared_blocks[i], 0, shared_data_space);
    #else
    net->p_shared_blocks[i] = (real*)malloc(shared_data_space);
    memset(net->p_shared_blocks[i], 0, shared_data_space);
    #endif
  }

  // temporary space at main memory
  //   max( temp_space, const_space, shared_data_space, param_data_space )
  {
    temp_cpu_space = MAX(net->temp_space,  net->const_space);
    temp_cpu_space = MAX(temp_cpu_space,  shared_data_space);
    temp_cpu_space = MAX(temp_cpu_space,  param_data_space);

    net->temp_cpu_data = (real*)malloc(temp_cpu_space);
    memset(net->temp_cpu_data, 0, temp_cpu_space);
  }

  // temporary space at GPU memory (in GPU mode)
  //   in CPU mode, temp_data referes to temp_cpu_data
  #ifdef GPU
  {
    cudaMalloc(&net->temp_data, net->temp_space);
    cudaMemset(net->temp_data, 0, net->temp_space);
  }
  #else
  {
    net->temp_space = temp_cpu_space;
    net->temp_data = net->temp_cpu_data;
  }
  #endif

  // space for constant vector [1, 1, ..., 1]
  #ifdef GPU
  {
    const int const_size = net->const_space / sizeof(real);
    for (int i = 0; i < const_size; ++i) {
      net->temp_cpu_data[i] = 1;
    }
    cudaMalloc(&net->const_data, net->const_space);
    cudaMemcpyAsync(net->const_data, net->temp_cpu_data, net->const_space,
                    cudaMemcpyHostToDevice);
  }
  #else
  {
    const int const_size = net->const_space / sizeof(real);
    net->const_data = (real*)malloc(net->const_space);
    for (int i = 0; i < const_size; ++i) {
      net->const_data[i] = 1;
    }
  }
  #endif

  // memory allocation for all tensors
  tensor_space = 0;
  for (int i = 0; i < net->num_tensors; ++i) {
    Tensor* const tensor = get_tensor(net, i);
    tensor_space +=
        malloc_tensor_data(tensor, net->p_shared_blocks[shared_block_id[i]]);

    // load pre-trained parameter data for parameter tensors
    if (tensor->data_type == PARAM_DATA) {
      char path[1024];
      sprintf(path, "%s/%s.bin", net->param_path, tensor->name);
      load_tensor_data(path, tensor, net->temp_cpu_data);
    }
  }

  // acquire CuBLAS handle
  #ifdef GPU
  {
    if (cublasCreate(&net->blas_handle) != CUBLAS_STATUS_SUCCESS) {
      printf("cublas creation failed\n");
    }
  }
  #endif

  // compute total space
  #ifdef GPU
  {
    net->space_cpu = sizeof(Net) + layer_cpu_space + temp_cpu_space;
    net->space = layer_space + net->num_shared_blocks * shared_data_space
                 + tensor_space + net->temp_space + net->const_space;
  }
  #else
  {
    net->space_cpu = sizeof(Net) + layer_cpu_space + temp_cpu_space
                     + net->num_shared_blocks * shared_data_space
                     + tensor_space + net->const_space;
    net->space = net->space_cpu;
  }
  #endif

  // set initialization flag
  net->initialized = 1;

  // summarization
  #ifdef GPU
  printf("%ldMB of main memory allocated\n",
         DIV_THEN_CEIL(net->space_cpu,  1000000));
  printf("  Net data structure at main memory = %ldKB\n",
         DIV_THEN_CEIL(sizeof(Net),  1000));
  printf("  Layer auxiliary data at main memory = %ldMB\n",
         DIV_THEN_CEIL(layer_cpu_space,  1000000));
  printf("  Temporary data at main memory = %ldMB\n",
         DIV_THEN_CEIL(temp_cpu_space,  1000000));
  printf("%ldMB of GPU memory allocated\n",
         DIV_THEN_CEIL(net->space,  1000000));
  #else
  printf("%ldMB of main memory allocated\n",
         DIV_THEN_CEIL(net->space_cpu,  1000000));
  printf("  Net data structure = %ldKB\n",
         DIV_THEN_CEIL(sizeof(Net),  1000));
  #endif
  printf("  Layer auxiliary data = %ldMB\n",
         DIV_THEN_CEIL(layer_space,  1000000));
  printf("  Shared tensor data = %ldMB\n",
       DIV_THEN_CEIL(net->num_shared_blocks * shared_data_space,  1000000));
  printf("  Private & parameter tensor data = %ldMB\n",
         DIV_THEN_CEIL(tensor_space,  1000000));
  printf("  Temporary data = %ldMB\n",
         DIV_THEN_CEIL(net->temp_space,  1000000));
  printf("  Constant vector data = %ldKB\n",
         DIV_THEN_CEIL(net->const_space,  1000));
}

void free_net(Net* const net)
{
  if (!net->initialized) {
    return;
  }

  for (int i = 0; i < net->num_layers; ++i) {
    Layer* const layer = get_layer(net, i);
    if (layer->f_free) {
      (*layer->f_free)(net, layer);
    }
  }

  for (int i = 0; i < net->num_shared_blocks; ++i) {
    #ifdef GPU
    cudaFree(net->p_shared_blocks[i]);
    #else
    free(net->p_shared_blocks[i]);
    #endif
  }

  for (int i = 0; i < net->num_tensors; ++i) {
    Tensor* const tensor = get_tensor(net, i);
    free_tensor_data(tensor);
  }

  free(net->temp_cpu_data);

  #ifdef GPU
  cudaFree(net->temp_data);
  cudaFree(net->const_data);
  #else
  free(net->const_data);
  #endif

  #ifdef GPU
  {
    if (cublasDestroy(net->blas_handle) != CUBLAS_STATUS_SUCCESS) {
      printf("cublas destruction failed\n");
    }
  }
  #endif

  memset(net, 0, sizeof(Net));
  free(net);
}

void save_layer_tops(void* const net_, void* const layer_)
{
  Net* const net = (Net*)net_;
  Layer* const layer = (Layer*)layer_;

  for (int i = 0; i < layer->num_tops; ++i) {
    char path[1024];
    const Tensor* const tensor = get_top(layer, i);
    sprintf(path, "%s/%s_top%d.rt.bin", net->param_path, layer->name, i);
    save_tensor_data(path, tensor, net->temp_cpu_data);
  }
}

void print_layer_tops(void* const net_, void* const layer_)
{
  const Net* const net = (Net*)net_;
  const Layer* const layer = (Layer*)layer_;

  for (int i = 0; i < layer->num_tops; ++i) {
    const Tensor* const tensor = get_top(layer, i);
    long int size = get_data_size(tensor);
    int idx[MAX_NDIM + 1] = { 0, };

    #ifdef GPU
    cudaMemcpyAsync(net->temp_cpu_data, tensor->data,
                    size * sizeof(real),
                    cudaMemcpyDeviceToHost);
    #else
    memcpy(net->temp_cpu_data, tensor->data, size * sizeof(real));
    #endif

    for (int j = 0; j < size; ++j) {
      const int n = idx[0];

      printf("Layer %s / Top %d / Image %d [", layer->name, i, n);
      for (int d = 1; d < tensor->ndim; ++d) {
        printf("%d, ", idx[d]);
      }
      printf("%d]: %f\n", idx[tensor->ndim]++, net->temp_cpu_data[j]);

      for (int d = tensor->ndim; d > 0; --d) {
        if (idx[d] == tensor->shape[n][d - 1]) {
          idx[d] = 0;
          ++idx[d - 1];
        }
      }
    }
  } // endfor i
}



// --------------------------------------------------------------------------
// simple functions returning static constants
//   required for Python interface
// --------------------------------------------------------------------------

int _max_num_tensors(void) { return MAX_NUM_TENSORS; }

int _max_num_layers(void) { return MAX_NUM_LAYERS; }

int _max_num_shared_blocks(void) { return MAX_NUM_SHARED_BLOCKS; }
