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

long int malloc_layer(Net* const net,
                      Layer* const layer)
{
  long int space = 0;

  // currently, do nothing for layer
  // aux_data is allocated by each operator's initializer

  #ifdef DEBUG
  printf("[Layer %s] %d bottoms, %d tops, %d params\n",
         layer->name,
         layer->num_bottoms, layer->num_tops, layer->num_params);
  #endif

  return space;
}

long int malloc_top_data(Net* const net,
                         Layer* const layer,
                         const int top_id)
{
  Tensor* const tensor = get_top(layer, top_id);
  long int space = 0;

  if (tensor->data_type == SHARED_DATA) {
    space = malloc_tensor_data(tensor, net->layer_data);
    net->space += space;
    printf("[Layer %s] malloc for top[%d], +%.2fKB\n",
           layer->name, top_id, (float)(space / 1000.0f));
  }

  return space;
}

long int free_top_data(Net* const net,
                       Layer* const layer,
                       const int top_id)
{
  Tensor* const tensor = get_top(layer, top_id);
  long int space = 0;

  if (tensor->data_type == SHARED_DATA) {
    space = free_tensor_data(tensor);
    tensor->data = net->layer_data[tensor->shared_block_id];
    net->space -= space;
    printf("[Layer %s] dealloc for top[%d], -%.2fKB\n",
           layer->name, top_id, (float)(space / 1000.0f));
  }

  return space;
}

void free_layer(Layer* const layer)
{
  for (int i = 0; i < layer->num_aux_data; ++i) {
    if (layer->p_aux_data[i]) {
      #ifdef GPU
      cudaFree(layer->p_aux_data[i]);
      #else
      free(layer->p_aux_data[i]);
      #endif
    }
    else {
      printf("[ERROR] Layer %s: memory for aux_data %d was not allocated\n",
             layer->name, i);
    }
  }

  memset(layer, 0, sizeof(Layer));
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

void assign_layer_data(Net* const net)
{
  int is_assigned[MAX_NUM_TENSORS] = { 0, };
  int alive_until[MAX_NUM_TENSORS] = { 0, };
  int reserved_until[MAX_NUM_LAYER_DATA] = { 0, };

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

  // assign layer_data to each tensor according to its lifetime
  for (int layer_id = 0; layer_id < net->num_layers; ++layer_id) {
    Layer* const layer = get_layer(net, layer_id);

    for (int top_id = 0; top_id < layer->num_tops; ++top_id) {
      Tensor* const tensor = get_top(layer, top_id);
      const int tensor_id = get_tensor_id(net, tensor);

      if (tensor->data_type == SHARED_DATA && !is_assigned[tensor_id]) {
        for (int block_id = 0; block_id < net->num_layer_data; ++block_id) {
          if (!reserved_until[block_id]) {
            tensor->shared_block_id = block_id;
            reserved_until[block_id] = alive_until[tensor_id];
            is_assigned[tensor_id] = 1;
            break;
          }
        }

        if (!is_assigned[tensor_id]) {
          if (net->num_layer_data == MAX_NUM_LAYER_DATA) {
            printf("[ERROR] Failed to assign layer_data for %s\n",
                   tensor->name);
          }
          else {
            const int block_id = net->num_layer_data++;
            tensor->shared_block_id = block_id;
            reserved_until[block_id] = alive_until[tensor_id];
            is_assigned[tensor_id] = 1;
          }          
        }
      }

      for (int block_id = 0; block_id < net->num_layer_data; ++block_id) {
        if (reserved_until[block_id] == layer_id) {
          reserved_until[block_id] = 0;
        }
      }
    }
  }

  for (int tensor_id = 0; tensor_id < net->num_tensors; ++tensor_id) {
    if (is_assigned[tensor_id]) {
      const Tensor* const tensor = get_tensor(net, tensor_id);
      printf("Tensor %s: Assigned shared memory block %d, ",
             tensor->name, tensor->shared_block_id);
      printf("reserved until layer %s\n",
             get_layer(net, alive_until[tensor_id])->name);
    }
  }
}

void init_net(Net* const net)
{
  memset(net, 0, sizeof(Net));
}

void malloc_net(Net* const net)
{
  long int space_cpu = 0;
  long int space = 0;

  {
  #ifdef GPU
    cudaMalloc(&net->temp_data, net->temp_size * sizeof(real));
    cudaMalloc(&net->tempint_data, net->tempint_size * sizeof(int));
    cudaMalloc(&net->const_data, net->const_size * sizeof(real));
  #else
    net->temp_data = (real*)malloc(net->temp_size * sizeof(real));
    net->tempint_data = (int*)malloc(net->tempint_size * sizeof(int));
    net->const_data = (real*)malloc(net->const_size * sizeof(real));
  #endif
    net->param_cpu_data = (real*)malloc(net->param_size * sizeof(real));
    net->temp_cpu_data = (real*)malloc(net->temp_size * sizeof(real));
    net->tempint_cpu_data = (int*)malloc(net->tempint_size * sizeof(int));
  }
  space += sizeof(real) * (net->temp_size + net->const_size)
           + sizeof(int) * (net->tempint_size);
  space_cpu += sizeof(real) * (2 * net->layer_size + net->param_size
                               + net->temp_size)
               + sizeof(int) * (net->tempint_size);

  // data initialization
  {
  #ifdef GPU
    for (int i = 0; i < net->const_size; ++i) {
      net->temp_cpu_data[i] = 1;
    }
    cudaMemcpyAsync(net->const_data, net->temp_cpu_data,
                    net->const_size * sizeof(real),
                    cudaMemcpyHostToDevice);
  #else
    for (int i = 0; i < net->const_size; ++i) {
      net->const_data[i] = 1;
    }
  #endif
  }

  // memory allocation for shared memory blocks
  assign_layer_data(net);
  for (int i = 0; i < net->num_layer_data; ++i) {
    #ifdef GPU
    cudaMalloc(&net->layer_data[i], net->layer_size * sizeof(real));
    #else
    net->layer_data[i] = (real*)malloc(net->layer_size * sizeof(real));
    #endif
  }
  space += net->num_layer_data * net->layer_size * sizeof(real);

  // memory allocation for tensors
  for (int i = 0; i < net->num_tensors; ++i) {
    Tensor* const tensor = get_tensor(net, i);
    const int tensor_size = malloc_tensor_data(tensor, net->layer_data);

    if (tensor->data_type == PARAM_DATA) {
      char path[1024];
      sprintf(path, "%s/%s.bin", net->param_path, tensor->name);
      load_tensor_data(path, tensor, net->param_cpu_data);
    }

    if (tensor->data_type == CPU_DATA) {
      space_cpu += tensor_size;
    }
    else {
      space += tensor_size;
    }
  }

  // memory allocation for layers
  for (int i = 0; i < net->num_layers; ++i) {
    Layer* const layer = get_layer(net, i);
    space += malloc_layer(net, layer);
  }
  space_cpu += net->num_layers * sizeof(Layer);

  // acquire CuBLAS handle
  #ifdef GPU
  {
    if (cublasCreate(&net->blas_handle) != CUBLAS_STATUS_SUCCESS) {
      printf("cublas creation failed\n");
    }
  }
  #endif

  net->space_cpu += space_cpu;
  net->space += space;

  net->initialized = 1;
}

void free_net(Net* const net)
{
  if (!net->initialized) {
    return;
  }

  {
  #ifdef GPU
    cudaFree(net->temp_data);
    cudaFree(net->tempint_data);
    cudaFree(net->const_data);
  #else
    free(net->temp_data);
    free(net->tempint_data);
    free(net->const_data);
  #endif
    free(net->param_cpu_data);
    free(net->temp_cpu_data);
    free(net->tempint_cpu_data);
  }

  for (int i = 0; i < net->num_layer_data; ++i) {
    #ifdef GPU
    cudaFree(net->layer_data[i]);
    #else
    free(net->layer_data[i]);
    #endif
  }

  for (int i = 0; i < net->num_tensors; ++i) {
    Tensor* const tensor = get_tensor(net, i);
    free_tensor_data(tensor);
  }

  for (int i = 0; i < net->num_layers; ++i) {
    Layer* const layer = get_layer(net, i);
    free_layer(layer);
  }

  #ifdef GPU
  {
    if (cublasDestroy(net->blas_handle) != CUBLAS_STATUS_SUCCESS) {
      printf("cublas destruction failed\n");
    }
  }
  #endif

  memset(net, 0, sizeof(Net));
}

void init_layers(Net* const net)
{
  for (int i = 0; i < net->num_layers; ++i) {
    Layer* const layer = get_layer(net, i);

    for (int j = 0; j < MAX_NUM_OPS_PER_LAYER; ++j) {
      if (layer->f_init[j]) {
        (*layer->f_init[j])(net, layer);
      }
    }
  }
}

void forward_net(Net* const net)
{
  for (int i = 0; i < net->num_layers; ++i) {
    Layer* const layer = get_layer(net, i);

    for (int j = 0; j < MAX_NUM_OPS_PER_LAYER; ++j) {
      if (layer->f_forward[j]) {
        (*layer->f_forward[j])(net, layer);
      }
    }
  }
}

void shape_net(Net* const net)
{
  for (int i = 0; i < net->num_layers; ++i) {
    Layer* const layer = get_layer(net, i);

    #ifdef DEBUG
    printf("[Layer %s]\n", layer->name);
    #endif

    for (int j = 0; j < MAX_NUM_OPS_PER_LAYER; ++j) {
      if (layer->f_shape[j]) {
        (*layer->f_shape[j])(net, layer);
        #ifdef DEBUG
        for (int top_id = 0; top_id < layer->num_tops; ++top_id) {
          const Tensor* const tensor = get_top(layer, top_id);
          print_tensor_info(tensor);
        }
        #endif
      }
    }
  }
}

void update_net_size(Net* const net,
                     const Layer* const layer,
                     const int temp_size,
                     const int tempint_size,
                     const int const_size)
{
  if (!net->initialized) {
    long int top_size = 0, param_size = 0;
    for (int i = 0; i < layer->num_tops; ++i) {
      const Tensor* const tensor = get_top(layer, i);
      if (tensor->data_type == SHARED_DATA) {
        top_size = MAX(top_size,  flatten_size(tensor));
      }
    }
    for (int i = 0; i < layer->num_params; ++i) {
      const Tensor* const tensor = get_param(layer, i);
      param_size = MAX(param_size,  flatten_size(tensor));
    }

    net->layer_size = MAX(net->layer_size,  top_size);
    net->param_size = MAX(net->param_size,  param_size);
    net->temp_size = MAX(net->temp_size,  (long)temp_size);
    net->tempint_size = MAX(net->tempint_size,  (long)tempint_size);
    net->const_size = MAX(net->const_size,  (long)const_size);
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
    long int size = flatten_size(tensor);
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
