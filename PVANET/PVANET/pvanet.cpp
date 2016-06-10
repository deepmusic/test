#include "layer.h"
#include "pvanet.hpp"
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

#include "logger.h"

static Net pvanet;
static bool initialized = false;

static const char* class_names[] = {
  "__background", "bicycle", "bird", "bus", "car", "cat", "dog", "horse",
  "motorbike", "person", "train", "aeroplane", "boat", "bottle", "chair",
  "cow", "diningtable", "pottedplant", "sheep", "sofa", "tvmonitor",
  "cake_choco", "cake_purple", "cake_white", "sportsbag"
};

void pvanet_init(const std::string& model_file, const std::string& weights_file, int gpu_id) {
  if (!initialized) {
    #ifdef GPU
    cudaSetDevice(gpu_id);
    #endif

    initialized = true;
    construct_pvanet(&pvanet, weights_file.c_str());
  }
}

void pvanet_release(void) {
  if (initialized) {
    free_net(&pvanet);
    initialized = false;
  }
}

void pvanet_detect(const unsigned char* image_data, int width, int height, int stride, std::vector<std::pair<std::string, std::vector<float> > >& boxes) {

    global_logger.start_log(PVANET_DETECT);

  boxes.clear();
  if (!initialized) {
      global_logger.stop_log(PVANET_DETECT);
      return;
  }

  process_pvanet(&pvanet, image_data, height, width, stride, NULL);

  {
    const real* const p_out_item = pvanet.output_cpu_data;
    for (int i = 0; i < pvanet.num_output_boxes; ++i) {
      std::pair<std::string, std::vector<float> > box;

      const int class_index = (int)p_out_item[i * 6 + 0];
      box.first = class_names[class_index];

      box.second.push_back(p_out_item[i * 6 + 1]);
      box.second.push_back(p_out_item[i * 6 + 2]);
      box.second.push_back(p_out_item[i * 6 + 3]);
      box.second.push_back(p_out_item[i * 6 + 4]);
      box.second.push_back(p_out_item[i * 6 + 5]);

      boxes.push_back(box);
    }
  }

  global_logger.stop_log(PVANET_DETECT);

  printf("%d objects has been detected.\n", pvanet.num_output_boxes);
}
