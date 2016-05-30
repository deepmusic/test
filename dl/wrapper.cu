#include "layer.h"

static Net pvanet;
static bool initialized = false;

static const char* class_names[] = {
  "__unknown__", "bicycle", "bird", "bus", "car", "cat", "dog", "horse",
  "motorbike", "person", "train", "aeroplane", "boat", "bottle", "chair",
  "cow", "diningtable", "pottedplant", "sheep", "sofa", "tvmonitor",
  "cake", "vase"
};

void _init_net(void)
{
  if (!initialized) {
    #ifdef GPU
    cudaSetDevice(0);
    #endif

    initialized = true;
    construct_pvanet(&pvanet, "scripts/params2");
  }
}

void _release_net(void)
{
  if (initialized) {
    free_net(&pvanet);
    initialized = false;
  }
}

void _detect_net(const unsigned char* const image_data,
                 const int width, const int height)
{
  if (!initialized) {
    return;
  }

  process_pvanet(&pvanet, image_data, height, width, NULL);
}
