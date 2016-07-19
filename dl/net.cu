#include "layer.h"
#include <string.h>

void init_layer(Layer* const layer)
{
  memset(layer, 0, sizeof(Layer));
}

void set_layer_name(Layer* const layer, const char* const name)
{
  strcpy(layer->name, name);
}

void set_bottom(Layer* const layer, const int bottom_id,
                Tensor* const tensor)
{
  if (bottom_id >= layer->num_bottoms) {
    printf("[ERROR] Layer %s: out-of-bound input index %d\n",
           layer->name, bottom_id);
    return;
  }
  layer->p_bottoms[bottom_id] = tensor;
}

void set_top(Layer* const layer, const int top_id,
             Tensor* const tensor)
{
  if (top_id >= layer->num_tops) {
    printf("[ERROR] Layer %s: out-of-bound output index %d\n",
           layer->name, top_id);
    return;
  }
  layer->p_tops[top_id] = tensor;
}

void set_param(Layer* const layer, const int param_id,
               Tensor* const tensor)
{
  if (param_id >= layer->num_params) {
    printf("[ERROR] Layer %s: out-of-bound parameter index %d\n",
           layer->name, param_id);
    return;
  }
  layer->p_params[param_id] = tensor;
  tensor->data_type = PARAM_DATA;
}

void add_bottom(Layer* const layer, Tensor* const tensor)
{
  if (layer->num_bottoms == MAX_NUM_BOTTOMS) {
    printf("[ERROR] Layer %s: cannot add more input\n", layer->name);
    for (int bottom_id = 0; bottom_id < layer->num_bottoms; ++bottom_id) {
      printf("  %s[%d]: %s\n",
             layer->name, bottom_id, get_bottom(layer, bottom_id)->name);
    }
    return;
  }
  ++layer->num_bottoms;
  set_bottom(layer, layer->num_bottoms - 1, tensor);
}

void add_top(Layer* const layer, Tensor* const tensor)
{
  if (layer->num_tops == MAX_NUM_TOPS) {
    printf("[ERROR] Layer %s: cannot add more output\n", layer->name);
    for (int top_id = 0; top_id < layer->num_tops; ++top_id) {
      printf("  %s[%d]: %s\n",
             layer->name, top_id, get_top(layer, top_id)->name);
    }
    return;
  }
  ++layer->num_tops;
  set_top(layer, layer->num_tops - 1, tensor);
}

void add_param(Layer* const layer, Tensor* const tensor)
{
  if (layer->num_params == MAX_NUM_PARAMS) {
    printf("[ERROR] Layer %s: cannot add more parameter\n", layer->name);
    for (int param_id = 0; param_id < layer->num_params; ++param_id) {
      printf("  %s[%d]: %s\n",
             layer->name, param_id, get_param(layer, param_id)->name);
    }
    return;
  }
  ++layer->num_params;
  set_param(layer, layer->num_params - 1, tensor);
}

Tensor* get_bottom(const Layer* const layer, const int bottom_id)
{
  #ifdef DEBUG
  if (bottom_id >= layer->num_bottoms) {
    printf("[ERROR] Layer %s: out-of-bound input index %d\n",
           layer->name, bottom_id);
    return NULL;
  }
  #endif
  return layer->p_bottoms[bottom_id];
}

Tensor* get_top(const Layer* const layer, const int top_id)
{
  #ifdef DEBUG
  if (top_id >= layer->num_tops) {
    printf("[ERROR] Layer %s: out-of-bound output index %d\n",
           layer->name, top_id);
    return NULL;
  }
  #endif
  return layer->p_tops[top_id];
}

Tensor* get_param(const Layer* const layer, const int param_id)
{
  #ifdef DEBUG
  if (param_id >= layer->num_params) {
    printf("[ERROR] Layer %s: out-of-bound parameter index %d\n",
           layer->name, param_id);
    return NULL;
  }
  #endif
  return layer->p_params[param_id];
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
    set_layer_name(layer, name);
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

  for (int tensor_id = 0; tensor_id < net->num_tensors; ++tensor_id) {
    if (is_assigned[tensor_id]) {
      const Tensor* const tensor = get_tensor(net, tensor_id);
      printf("Tensor %s: Assigned shared memory block %d, ",
             tensor->name, shared_block_id[tensor_id]);
      printf("reserved until layer %s\n",
             get_layer(net, alive_until[tensor_id])->name);
    }
  }
}

void init_net(Net* const net)
{
  memset(net, 0, sizeof(Net));
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

void malloc_net_data(Net* const net)
{
  if (!net->initialized) {
    long int shared_data_space = 0;
    long int param_data_space = 0;
    long int private_data_space = 0;

    for (int i = 0; i < net->num_tensors; ++i) {
      const Tensor* const tensor = get_tensor(net, i);
      if (tensor->data_type == SHARED_DATA) {
        shared_data_space = MAX(shared_data_space,  get_data_size(tensor));
      }
      else if (tensor->data_type == PARAM_DATA) {
        param_data_space = MAX(param_data_space,  get_data_size(tensor));
      }
      else {
        private_data_space = MAX(private_data_space,  get_data_size(tensor));
      }
    }
    shared_data_space *= sizeof(real);
    param_data_space *= sizeof(real);
    private_data_space *= sizeof(real);

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
    net->space += net->num_shared_blocks * shared_data_space;
    printf("Shared data space = %ldMB\n",
      DIV_THEN_CEIL(net->num_shared_blocks * shared_data_space,  1000000));

    // temporary space at main memory
    {
      long int temp_cpu_space = MAX(net->temp_space,  net->const_space);
      temp_cpu_space = MAX(temp_cpu_space,  shared_data_space);
      temp_cpu_space = MAX(temp_cpu_space,  param_data_space);

      net->temp_cpu_data = (real*)malloc(temp_cpu_space);
      memset(net->temp_cpu_data, 0, temp_cpu_space);

      net->space_cpu += temp_cpu_space;
      printf("Temporary space = %ldMB\n",
             DIV_THEN_CEIL(temp_cpu_space,  1000000));
    }

    // temporary space at GPU memory (in GPU mode)
    //   in CPU mode, temp_data referes to temp_cpu_data
    #ifdef GPU
    {
      cudaMalloc(&net->temp_data, net->temp_space);
      cudaMemset(net->temp_data, 0, net->temp_space);

      net->space += net->temp_space;
      printf("Temporary space at GPU = %ldMB\n",
             DIV_THEN_CEIL(net->temp_space,  1000000));
    }
    #else
    net->temp_data = net->temp_cpu_data;
    #endif

    // space for constant vector
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
    net->space += net->const_space;
    printf("Const space = %ldKB\n",
           DIV_THEN_CEIL(net->const_space,  1000));
  }
}

void malloc_net(Net* const net)
{
  long int space = 0;

  int shared_block_id[MAX_NUM_TENSORS] = { 0, };
  assign_shared_blocks(net, shared_block_id);

  malloc_net_data(net);

  for (int i = 0; i < net->num_tensors; ++i) {
    Tensor* const tensor = get_tensor(net, i);
    space += malloc_tensor_data(tensor,
        net->p_shared_blocks[shared_block_id[i]]);

    if (tensor->data_type == PARAM_DATA) {
      char path[1024];
      sprintf(path, "%s/%s.bin", net->param_path, tensor->name);
      load_tensor_data(path, tensor, net->temp_cpu_data);
    }
  }
  net->space += space;
  printf("Private & param data space = %ldMB\n",
         DIV_THEN_CEIL(space,  1000000));

  // acquire CuBLAS handle
  #ifdef GPU
  {
    if (cublasCreate(&net->blas_handle) != CUBLAS_STATUS_SUCCESS) {
      printf("cublas creation failed\n");
    }
  }
  #endif

  net->space_cpu += sizeof(Net);
  net->initialized = 1;

  // print total memory size required
  {
  #ifdef GPU
    printf("%ldMB of main memory allocated\n",
           DIV_THEN_CEIL(net->space_cpu,  1000000));
    printf("%ldMB of GPU memory allocated\n",
           DIV_THEN_CEIL(net->space,  1000000));
  #else
    printf("%ldMB of main memory allocated\n",
           DIV_THEN_CEIL(net->space + net->space_cpu,  1000000));
  #endif
  }
}

void free_net(Net* const net)
{
  if (!net->initialized) {
    return;
  }

  {
  #ifdef GPU
    cudaFree(net->temp_data);
    cudaFree(net->const_data);
  #else
    free(net->const_data);
  #endif
    free(net->temp_cpu_data);
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

  for (int i = 0; i < net->num_layers; ++i) {
    Layer* const layer = get_layer(net, i);
    if (layer->f_free) {
      (*layer->f_free)(net, layer);
    }
    init_layer(layer);
  }

  #ifdef GPU
  {
    if (cublasDestroy(net->blas_handle) != CUBLAS_STATUS_SUCCESS) {
      printf("cublas destruction failed\n");
    }
  }
  #endif

  init_net(net);
}

void forward_net(Net* const net)
{
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

    if (size > 1000) {
      size = 1000;
    }

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
