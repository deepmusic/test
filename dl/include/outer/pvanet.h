#ifndef PVA_DL_PVANET_H
#define PVA_DL_PVANET_H

#include "nets/net_factory.h"

// --------------------------------------------------------------------------
// PVANET interface
//   callable from external environments via shared library
// --------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

Net* create_pvanet(const char* const param_path,
                   const int is_light_model,
                   const int fc_compress,
                   const int pre_nms_topn,
                   const int post_nms_topn,
                   const int input_size);

void process_pvanet(Net* const net,
                    unsigned char image[],
                    const int image_height,
                    const int image_width,
                    real** const ref_output_boxes,
                    int* const p_num_boxes,
                    FILE* fp);

void process_batch_pvanet(Net* const net,
                          unsigned char* const p_images[],
                          const int image_heights[],
                          const int image_widths[],
                          const int num_images,
                          real** const ref_output_boxes,
                          int num_boxes[],
                          FILE* fp);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end PVA_DL_PVANET_H
