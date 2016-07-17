#ifndef PVA_DL_PVANET_HPP
#define PVA_DL_PVANET_HPP

#include <string>
#include <vector>

extern "C" {

void pvanet_init(const std::string& model_file, const std::string& weights_file, int gpu_id);
void pvanet_release();
void pvanet_detect(const unsigned char* image_data, int width, int height, int stride, std::vector<std::pair<std::string, std::vector<float> > >& boxes);

}

#endif // end PVA_DL_PVANET_HPP
