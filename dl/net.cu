#include "layer.h"
#include <string.h>
#include <stdio.h>

long int malloc_layer(Layer* const layer)
{
  long int space_cpu = 0;

  layer->p_bottoms = (layer->num_bottoms <= 0) ? NULL :
                     (Tensor**)malloc(layer->num_bottoms * sizeof(Tensor*));
  space_cpu += layer->num_bottoms * sizeof(Tensor*);

  layer->tops = (layer->num_tops <= 0) ? NULL :
                (Tensor*)malloc(layer->num_tops * sizeof(Tensor));
  layer->allocate_top_data = (layer->num_tops <= 0) ? NULL:
                             (int*)calloc(layer->num_tops, sizeof(int));
  space_cpu += layer->num_tops * (sizeof(Tensor) + sizeof(int));

  layer->params = (layer->num_params <= 0) ? NULL : 
                  (Tensor*)malloc(layer->num_params * sizeof(Tensor));
  space_cpu += layer->num_params * sizeof(Tensor);

  return space_cpu;
}

long int malloc_load_layer_data(Layer* const layer,
                                const char* const name,
                                real* const temp_cpu_space)
{
  long int space = 0;

  for (int i = 0; i < layer->num_tops; ++i) {
    if (layer->allocate_top_data[i]) {
      space += malloc_tensor(&layer->tops[i]);
    }
  }

  for (int i = 0; i < layer->num_params; ++i) {
    char path[1024];
    //printf("malloc param %d\n", i);
    space += malloc_tensor(&layer->params[i]);
    sprintf(path, "params/%s_param%d.bin", name, i);
    //printf("load param %s\n", path);
    load_tensor(path, &layer->params[i], temp_cpu_space);
  }

  return space;
}

void free_layer(Layer* const layer)
{
  if (layer->p_bottoms) {
    free(layer->p_bottoms);
    layer->p_bottoms = NULL;
  }

  if (layer->tops) {
    for (int i = 0; i < layer->num_tops; ++i) {
      if (layer->allocate_top_data[i]) {
        #ifdef GPU
        cudaFree(layer->tops[i].data);
        layer->tops[i].data = NULL;
        #else
        free(layer->tops[i].data);
        layer->tops[i].data = NULL;
        #endif
      }
    }
    free(layer->tops);
    layer->tops = NULL;
    free(layer->allocate_top_data);
    layer->allocate_top_data = NULL;
  }

  if (layer->params) {
    for (int i = 0; i < layer->num_params; ++i) {
      #ifdef GPU
      cudaFree(layer->params[i].data);
      layer->params[i].data = NULL;
      #else
      free(layer->params[i].data);
      layer->params[i].data = NULL;
      #endif
    }
    free(layer->params);
    layer->params = NULL;
  }

  free(layer);
}

void malloc_net(Net* const net)
{
  long int space_cpu = 0;
  long int space = 0;

  for (int i = 0; i < net->num_layer_data; ++i) {
    #ifdef GPU
    cudaMalloc(&net->layer_data[i], net->layer_size * sizeof(real));
    #else
    net->layer_data[i] = (real*)malloc(net->layer_size * sizeof(real));
    #endif
    net->reserved_layer_data[i] = 0;
  }
  space += net->num_layer_data * net->layer_size * sizeof(real);

  #ifdef GPU
  {
    cudaMalloc(&net->temp_data, net->temp_size * sizeof(real));
    cudaMalloc(&net->tempint_data, net->tempint_size * sizeof(int));
    cudaMalloc(&net->const_data, net->const_size * sizeof(real));
/*
    cudaMallocHost(&net->input_cpu_data, net->layer_size * sizeof(real));
    cudaMallocHost(&net->output_cpu_data, net->layer_size * sizeof(real));
    cudaMallocHost(&net->param_cpu_data, net->param_size * sizeof(real));
    cudaMallocHost(&net->temp_cpu_data, net->temp_size * sizeof(real));
    cudaMallocHost(&net->tempint_cpu_data, net->tempint_size * sizeof(int));
*/
    net->input_cpu_data = (real*)malloc(net->layer_size * sizeof(real));
    net->output_cpu_data = (real*)malloc(net->layer_size * sizeof(real));
    net->param_cpu_data = (real*)malloc(net->param_size * sizeof(real));
    net->temp_cpu_data = (real*)malloc(net->temp_size * sizeof(real));
    net->tempint_cpu_data = (int*)malloc(net->tempint_size * sizeof(int));
  }
  #else
  {
    net->temp_data = (real*)malloc(net->temp_size * sizeof(real));
    net->tempint_data = (int*)malloc(net->tempint_size * sizeof(int));
    net->const_data = (real*)malloc(net->const_size * sizeof(real));

    net->input_cpu_data = (real*)malloc(net->layer_size * sizeof(real));
    net->output_cpu_data = (real*)malloc(net->layer_size * sizeof(real));
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
      net->output_cpu_data[i] = 1;
    }
    cudaMemcpyAsync(net->const_data, net->output_cpu_data,
                    net->const_size * sizeof(real),
                    cudaMemcpyHostToDevice);
  #else
    for (int i = 0; i < net->const_size; ++i) {
      net->const_data[i] = 1;
    }
  #endif
  }

  for (int i = 0; i < net->num_layers; ++i) {
    space += malloc_load_layer_data(net->layers[i], net->layers[i]->name,
                                    net->param_cpu_data);
  }

  net->img_info->data
      = (real*)malloc(flatten_size(net->img_info) * sizeof(real));
  space_cpu += sizeof(real) * flatten_size(net->img_info);

  // acquire CuBLAS handle
  #ifdef GPU
  {
    if (cublasCreate(&net->cublas_handle) != CUBLAS_STATUS_SUCCESS) {
      printf("cublas creation failed\n");
    }
  }
  #endif

  net->space_cpu = space_cpu;
  net->space = space;

  net->initialized = 1;
}

void free_net(Net* const net)
{
  for (int i = 0; i < net->num_layers; ++i) {
    free_layer(net->layers[i]);
  }

  for (int i = 0; i < net->num_layer_data; ++i) {
    #ifdef GPU
    cudaFree(net->layer_data[i]);
    #else
    free(net->layer_data[i]);
    #endif
    net->layer_data[i] = NULL;
  }

  #ifdef GPU
  {
    cudaFree(net->temp_data);
    cudaFree(net->tempint_data);
    cudaFree(net->const_data);
    cudaFree(net->anchors);

    cudaFreeHost(net->input_cpu_data);
    cudaFreeHost(net->output_cpu_data);
    cudaFreeHost(net->param_cpu_data);
    cudaFreeHost(net->temp_cpu_data);
    cudaFreeHost(net->tempint_cpu_data);
  }
  #else
  {
    free(net->temp_data);
    free(net->tempint_data);
    free(net->const_data);
    free(net->anchors);

    free(net->input_cpu_data);
    free(net->output_cpu_data);
    free(net->param_cpu_data);
    free(net->temp_cpu_data);
    free(net->tempint_cpu_data);
  }
  #endif


  free(net->img_info->data);
  free(net->img_info);

  net->temp_data = NULL;
  net->tempint_data = NULL;
  net->const_data = NULL;
  net->input_cpu_data = NULL;
  net->output_cpu_data = NULL;
  net->param_cpu_data = NULL;
  net->temp_cpu_data = NULL;
  net->tempint_cpu_data = NULL;
  net->anchors = NULL;
  net->img_info = NULL;

  #ifdef GPU
  {
    if (cublasDestroy(net->cublas_handle) != CUBLAS_STATUS_SUCCESS) {
      printf("cublas destruction failed\n");
    }
  }
  #endif

  net->initialized = 0;
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
      if (!layer->allocate_top_data[i]) {
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

real* get_layer_data(Net* const net)
{
  for (int i = 0; i < net->num_layer_data; ++i) {
    if (!net->reserved_layer_data[i]) {
      net->reserved_layer_data[i] = 1;
      return net->layer_data[i];
    }
  }

  printf("[ERROR] Not enough temporary space for storing layer output!\n");
  return NULL;
}

void print_layer_tops(const Net* const net,
                      const Layer* const layer)
{
  for (int i = 0; i < layer->num_tops; ++i) {
    const long int size = flatten_size(&layer->tops[i]);
    #ifdef GPU
    cudaMemcpyAsync(net->output_cpu_data, layer->tops[i].data,
                    size * sizeof(real),
                    cudaMemcpyDeviceToHost);
    #else
    memcpy(net->output_cpu_data, layer->tops[i].data, size * sizeof(real));
    #endif
    char path[1024];
    sprintf(path, "params/%s_top%d.txt", layer->name, i);
    FILE* fp = fopen(path, "w");
    const Tensor* const t = &layer->tops[0];
    int j = 0;
    for (int n = 0; n < t->num_items; ++n) {
      for (int c = 0; c < t->shape[n][0]; ++c)
        for (int h = 0; h < t->shape[n][1]; ++h)
          for (int w = 0; w < t->shape[n][2]; ++w)
            fprintf(fp, "%d %d %d %d %f\n",
                    n, c, h, w, net->output_cpu_data[j++]);
    }
    fclose(fp);
  }
}
