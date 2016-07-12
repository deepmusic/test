#include "layer.h"
#include <string.h>

static Net pvanet;
static int initialized = 0;

int _batch_size_net(void)
{
  return BATCH_SIZE;
}
int _max_num_bottoms(void)
{
  return MAX_NUM_BOTTOMS;
}
int _max_num_tops(void)
{
  return MAX_NUM_TOPS;
}
int _max_num_params(void)
{
  return MAX_NUM_PARAMS;
}
int _max_num_auxs(void)
{
  return MAX_NUM_AUXS;
}
int _max_num_ops_per_layer(void)
{
  return MAX_NUM_OPS_PER_LAYER;
}

void _generate_net(void)
{
  if (!initialized) {
    #ifdef GPU
    cudaSetDevice(0);
    #endif

    initialized = 1;
    construct_pvanet(&pvanet, "scripts/params3");
  }
  else {
    printf("[ERROR] Release the current network first\n");
  }
}

void _init_net(void)
{
  if (!initialized) {
    init_net(&pvanet);
  }
  else {
    printf("[ERROR] Release the current network first\n");
  }
}

void _set_net_name(const char* const param_path)
{
  if (!initialized) {
    strcpy(pvanet.param_path, param_path);
  }
  else {
    printf("[ERROR] Release the current network first\n");
  }
}

void _shape_net(void)
{
  shape_net(&pvanet);
}

void _malloc_net(void)
{
  if (!initialized) {
    malloc_net(&pvanet);
    initialized = 1;
  }
  else {
    printf("[ERROR] Release the current network first\n");
  }
}

void _init_layers(void)
{
  if (initialized) {
    init_layers(&pvanet);
  }
  else {
    printf("[ERROR] Create a network instance first\n");
  }
}

void _release_net(void)
{
  if (initialized) {
    free_net(&pvanet);
    initialized = 0;
  }
  else {
    printf("[ERROR] Create a network instance first\n");
  }
}

void _detect_net(const unsigned char* const image_data,
                 const int width, const int height)
{
  if (initialized) {
    process_pvanet(&pvanet, image_data, height, width, NULL);
  }
  else {
    printf("[ERROR] Create a network instance first\n");
  }
}

Tensor* _layer_net(const int layer_id, const int top_id)
{
  if (layer_id >= 0 && layer_id < pvanet.num_layers &&
      top_id >= 0 && top_id < pvanet.layers[layer_id].num_tops)
  {
    return pvanet.layers[layer_id].p_tops[top_id];
  }

  return NULL;
}

void _print_layer(const int layer_id)
{
  if (layer_id >= 0 && layer_id < pvanet.num_layers) {
    Layer* const layer = &pvanet.layers[layer_id];

    for (int i = 0; i < MAX_NUM_OPS_PER_LAYER; ++i) {
      if (layer->f_forward[i] == print_layer_tops) {
        printf("[Layer %d (%s)]: Logging OFF\n", layer_id, layer->name);
        layer->f_forward[i] = NULL;
        for (int j = 0; j < layer->num_tops; ++j) {
          free_top_data(&pvanet, layer, j);
        }
        return;
      }

      if (!layer->f_forward[i]) {
        printf("[Layer %d (%s)]: Logging ON\n", layer_id, layer->name);
        layer->f_forward[i] = print_layer_tops;
        for (int j = 0; j < layer->num_tops; ++j) {
          malloc_top_data(&pvanet, layer, j);
        }
        return;
      }
    }

    printf("[Layer %d (%s)]: Can't add anymore operator\n",
           layer_id, layer->name);
  }
}
