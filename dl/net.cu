#include "layer.h"
#include <string.h>

void init_layer(Layer* const layer)
{
  memset(layer, 0, sizeof(Layer));
}

void set_layer_name(Layer* const layer, const char* const name)
{
  strcpy(layer->name, name);
  for (int i = 0; i < layer->num_tops; ++i) {
    sprintf(layer->tops[i].name, "%s[%d]", name, i);
  }
  for (int i = 0; i < layer->num_params; ++i) {
    sprintf(layer->params[i].name, "%s_param%d", name, i);
  }
}

long int malloc_layer(Net* const net,
                      Layer* const layer)
{
  long int space = 0;

  #ifdef DEBUG
  printf("%s %d %d\n", layer->name, layer->num_tops, layer->num_params);
  #endif

  for (int i = 0; i < layer->num_tops; ++i) {
    Tensor* const tensor = &layer->tops[i];

    tensor->max_data_size = flatten_size(tensor);

    if (tensor->has_own_memory) {
      space += malloc_tensor_data(tensor);
    }
    else if (tensor->data_id > 0) {
      tensor->data = net->layer_data[tensor->data_id - 1];
    }
    else {
      printf("[ERROR] Wrong data id %d for layer %s[%d]\n",
             tensor->data_id, layer->name, i);
    }
  }

  for (int i = 0; i < layer->num_params; ++i) {
    char path[1024];
    Tensor* const tensor = &layer->params[i];
    tensor->max_data_size = flatten_size(tensor);
    space += malloc_tensor_data(tensor);
    sprintf(path, "%s/%s.bin", net->param_path, tensor->name);
    load_tensor(path, tensor, net->temp_cpu_data);
  }

  return space;
}

long int malloc_top_data(Net* const net,
                         Layer* const layer,
                         const int top_id)
{
  Tensor* const tensor = &layer->tops[top_id];
  long int space = 0;

  if (!tensor->has_own_memory && tensor->data_id > 0) {
    tensor->has_own_memory = 1;
    tensor->data = NULL;
    space = malloc_tensor_data(tensor);
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
  Tensor* const tensor = &layer->tops[top_id];
  long int space = 0;

  if (tensor->has_own_memory && tensor->data_id > 0) {
    tensor->has_own_memory = 0;
    space = free_tensor_data(tensor);
    tensor->data = net->layer_data[tensor->data_id - 1];
    net->space -= space;
    printf("[Layer %s] dealloc for top[%d], -%.2fKB\n",
           layer->name, top_id, (float)(space / 1000.0f));
  }

  return space;
}

void free_layer(Layer* const layer)
{
  for (int i = 0; i < layer->num_tops; ++i) {
    if (layer->tops[i].has_own_memory) {
      free_tensor_data(&layer->tops[i]);
    }
  }

  for (int i = 0; i < layer->num_params; ++i) {
    free_tensor_data(&layer->params[i]);
  }

  for (int i = 0; i < layer->num_aux_data; ++i) {
    #ifdef GPU
    cudaFree(layer->p_aux_data[i]);
    #else
    free(layer->p_aux_data[i]);
    #endif
  }

  memset(layer, 0, sizeof(Layer));
}

void assign_layer_data(Net* const net)
{
  // compute lifetime for each tensor
  for (int layer_id = net->num_layers - 1; layer_id >= 0; --layer_id) {
    Layer* const layer = &net->layers[layer_id];
    for (int bottom_id = 0; bottom_id < layer->num_bottoms; ++bottom_id) {
      if (!layer->p_bottoms[bottom_id]->alive_until) {
        layer->p_bottoms[bottom_id]->alive_until = (void*)layer;
      }
    }
  }

  // lifetime for output tensors
  for (int layer_id = 0; layer_id < net->num_layers; ++layer_id) {
    Layer* const layer = &net->layers[layer_id];
    for (int top_id = 0; top_id < layer->num_tops; ++top_id) {
      Tensor* const tensor = &layer->tops[top_id];
      if (!tensor->alive_until) {
        tensor->alive_until = (void*)&net->layers[net->num_layers - 1];
      }
    }
  }

  // assign layer_data to each tensor according to its lifetime
  for (int layer_id = 0; layer_id < net->num_layers; ++layer_id) {
    Layer* const layer = &net->layers[layer_id];

    for (int top_id = 0; top_id < layer->num_tops; ++top_id) {
      Tensor* const tensor = &layer->tops[top_id];

      if (!tensor->has_own_memory) {
        for (int data_id = 0; data_id < net->num_layer_data; ++data_id) {
          if (!net->reserved_until[data_id]) {
            tensor->data_id = data_id + 1;
            net->reserved_until[data_id] = tensor->alive_until;
            printf("%s: assigned layer_data[%d], reserved until %s\n",
                   tensor->name, data_id,
                   ((Layer*)tensor->alive_until)->name);
            break;
          }
        }
        if (!tensor->data_id) {
          printf("[ERROR] Failed to assign layer_data for %s\n",
                 tensor->name);
        }
      }

      for (int data_id = 0; data_id < net->num_layer_data; ++data_id) {
        if (net->reserved_until[data_id] == (void*)layer) {
          net->reserved_until[data_id] = NULL;
        }
      }
    }
  }
}

void init_net(Net* const net)
{
  memset(net, 0, sizeof(Net));
  for (int i = 0; i < MAX_NUM_LAYERS; ++i) {
    init_layer(&net->layers[i]);
  }
}

void malloc_net(Net* const net)
{
  long int space_cpu = 0;
  long int space = 0;

  space_cpu += net->num_layers * sizeof(Layer);

  for (int i = 0; i < net->num_layer_data; ++i) {
    #ifdef GPU
    cudaMalloc(&net->layer_data[i], net->layer_size * sizeof(real));
    #else
    net->layer_data[i] = (real*)malloc(net->layer_size * sizeof(real));
    #endif
  }
  space += net->num_layer_data * net->layer_size * sizeof(real);

  #ifdef GPU
  {
    cudaMalloc(&net->temp_data, net->temp_size * sizeof(real));
    cudaMalloc(&net->tempint_data, net->tempint_size * sizeof(int));
    cudaMalloc(&net->const_data, net->const_size * sizeof(real));

    net->param_cpu_data = (real*)malloc(net->param_size * sizeof(real));
    net->temp_cpu_data = (real*)malloc(net->temp_size * sizeof(real));
    net->tempint_cpu_data = (int*)malloc(net->tempint_size * sizeof(int));
  }
  #else
  {
    net->temp_data = (real*)malloc(net->temp_size * sizeof(real));
    net->tempint_data = (int*)malloc(net->tempint_size * sizeof(int));
    net->const_data = (real*)malloc(net->const_size * sizeof(real));

    net->param_cpu_data = (real*)malloc(net->param_size * sizeof(real));
    net->temp_cpu_data = (real*)malloc(net->temp_size * sizeof(real));
    net->tempint_cpu_data = (int*)malloc(net->tempint_size * sizeof(int));
  }
  #endif
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

  assign_layer_data(net);

  // memory allocation for layers
  for (int i = 0; i < net->num_layers; ++i) {
    space += malloc_layer(net, &net->layers[i]);
  }

  {
    Tensor* img_info = &net->img_info;
    const long int img_info_size = flatten_size(img_info);
    img_info->data = (real*)malloc(img_info_size * sizeof(real));
    space_cpu += sizeof(real) * img_info_size;
  }

  // acquire CuBLAS handle
  #ifdef GPU
  {
    if (cublasCreate(&net->cublas_handle) != CUBLAS_STATUS_SUCCESS) {
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
  for (int i = 0; i < net->num_layers; ++i) {
    free_layer(&net->layers[i]);
  }

  for (int i = 0; i < net->num_layer_data; ++i) {
    #ifdef GPU
    cudaFree(net->layer_data[i]);
    #else
    free(net->layer_data[i]);
    #endif
  }

  #ifdef GPU
  {
    cudaFree(net->temp_data);
    cudaFree(net->tempint_data);
    cudaFree(net->const_data);

    free(net->param_cpu_data);
    free(net->temp_cpu_data);
    free(net->tempint_cpu_data);
  }
  #else
  {
    free(net->temp_data);
    free(net->tempint_data);
    free(net->const_data);

    free(net->param_cpu_data);
    free(net->temp_cpu_data);
    free(net->tempint_cpu_data);
  }
  #endif

  free(net->img_info.data);

  #ifdef GPU
  {
    if (cublasDestroy(net->cublas_handle) != CUBLAS_STATUS_SUCCESS) {
      printf("cublas destruction failed\n");
    }
  }
  #endif

  memset(net, 0, sizeof(Net));
}

void init_layers(Net* const net)
{
  for (int i = 0; i < net->num_layers; ++i) {
    Layer* const layer = &net->layers[i];

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
    Layer* const layer = &net->layers[i];

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
    Layer* const layer = &net->layers[i];

    for (int j = 0; j < MAX_NUM_OPS_PER_LAYER; ++j) {
      if (layer->f_shape[j]) {
        (*layer->f_shape[j])(net, layer);
        #ifdef DEBUG
        for (int k = 0; k < layer->num_tops; ++k) {
          print_tensor_info(layer->name, &layer->tops[k]);
        }
        #endif
      }
    }
  }
}

void _assign_layer_data(Net* const net, Tensor* const tensor)
{
  if (tensor->data_id) {
    printf("[WARNING] Reallocate layer_data for %s\n", tensor->name);
  }

  for (int i = 0; i < MAX_NUM_LAYER_DATA; ++i) {
    if (!net->reserved_until[i]) {
      tensor->data_id = i + 1;
      net->reserved_until[i] = (void*)1;
      #ifdef DEBUG
      printf("%s: assigned layer_data[%d]\n", tensor->name, i);
      #endif
      return;
    }
  }
  printf("[ERROR] Failed to assign layer_data for %s\n", tensor->name);
}

void _deallocate_layer_data(Net* const net, Tensor* const tensor)
{
  if (tensor->data_id) {
    net->reserved_until[tensor->data_id - 1] = 0;
    #ifdef DEBUG
    printf("%s: deallocated layer_data[%d]\n",
           tensor->name, tensor->data_id - 1);
    #endif
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
      if (!layer->tops[i].has_own_memory) {
        top_size = MAX(top_size,  flatten_size(&layer->tops[i]));
      }
    }
    for (int i = 0; i < layer->num_params; ++i) {
      param_size = MAX(param_size,  flatten_size(&layer->params[i]));
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
    sprintf(path, "%s/%s_top%d.rt.bin", net->param_path, layer->name, i);
    save_tensor_data(path, &layer->tops[i], net->temp_cpu_data);
  }
}

void print_layer_tops(void* const net_, void* const layer_)
{
  const Net* const net = (Net*)net_;
  const Layer* const layer = (Layer*)layer_;

  for (int i = 0; i < layer->num_tops; ++i) {
    const long int size = flatten_size(&layer->tops[i]);
    const Tensor* const t = &layer->tops[i];
    int idx[MAX_NDIM + 1] = { 0, };

    #ifdef GPU
    cudaMemcpyAsync(net->temp_cpu_data, layer->tops[i].data,
                    size * sizeof(real),
                    cudaMemcpyDeviceToHost);
    #else
    memcpy(net->temp_cpu_data, layer->tops[i].data, size * sizeof(real));
    #endif

    for (int j = 0; j < size; ++j) {
      const int n = idx[0];

      printf("Layer %s / Top %d / Image %d [", layer->name, i, n);
      for (int d = 1; d < t->ndim; ++d) {
        printf("%d, ", idx[d]);
      }
      printf("%d]: %f\n", idx[t->ndim]++, net->temp_cpu_data[j]);

      for (int d = t->ndim; d > 0; --d) {
        if (idx[d] == t->shape[n][d - 1]) {
          idx[d] = 0;
          ++idx[d - 1];
        }
      }
    }
  } // endfor i
}
