#include "net.h"
#include "pvanet.hpp"
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

void pvanet_init(const std::string& model_file, const std::string& weights_file, int gpu_id) {
  pvanet_7_1_1_init(weights_file.c_str(), gpu_id);
}

void pvanet_release(void) {
  pvanet_7_1_1_release();
}

void pvanet_detect(const unsigned char* image_data, int width, int height, int stride, std::vector<std::pair<std::string, std::vector<float> > >& boxes) {
  pvanet_7_1_1_detect(image_data, width, height, stride, boxes);
}
