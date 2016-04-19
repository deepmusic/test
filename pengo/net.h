#ifndef PVANET_NET_H
#define PVANET_NET_H

#include <string>
#include <vector>

void pvanet_7_1_1_init(const char* const param_path, int gpu_id);
void pvanet_7_1_1_release(void);
void pvanet_7_1_1_detect(const unsigned char* image_data, int width, int height, int stride, std::vector<std::pair<std::string, std::vector<float> > >& boxes);

#endif
