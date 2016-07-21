#include "core/layer.h"

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



// --------------------------------------------------------------------------
// simple functions returning static constants
//   required for Python interface
// --------------------------------------------------------------------------

int _max_num_bottoms(void) { return MAX_NUM_BOTTOMS; }

int _max_num_tops(void) { return MAX_NUM_TOPS; }

int _max_num_params(void) { return MAX_NUM_PARAMS; }
