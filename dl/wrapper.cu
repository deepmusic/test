#include "layer.h"
#include <string.h>

static Net pvanet;

int _batch_size_net(void)
{
  return BATCH_SIZE;
}
int _max_ndim(void)
{
  return MAX_NDIM;
}
int _max_name_len(void)
{
  return MAX_NAME_LEN;
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

int _max_num_tensors(void)
{
  return MAX_NUM_TENSORS;
}
int _max_num_layers(void)
{
  return MAX_NUM_LAYERS;
}
int _max_num_shared_blocks(void)
{
  return MAX_NUM_SHARED_BLOCKS;
}

Net* _net(void)
{
  return &pvanet;
}

void _init_net(void)
{
  if (!pvanet.initialized) {
    init_net(&pvanet);
  }
  else {
    printf("[ERROR] Release the current network first\n");
  }
}

void _set_net_param_path(const char* const param_path)
{
  if (!pvanet.initialized) {
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
  if (!pvanet.initialized) {
    malloc_net(&pvanet);
  }
  else {
    printf("[ERROR] Release the current network first\n");
  }
}

void _release_net(void)
{
  if (pvanet.initialized) {
    free_net(&pvanet);
  }
  else {
    printf("[ERROR] Create a network instance first\n");
  }
}

void _detect_net(const unsigned char image_data[],
                 const int width, const int height)
{
  if (pvanet.initialized) {
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
