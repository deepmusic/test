#ifndef PVA_DL_PVANET_H
#define PVA_DL_PVANET_H

#include "net.h"

// --------------------------------------------------------------------------
// network creators
// --------------------------------------------------------------------------

void setup_shared_cnn(Net* const net);

void setup_shared_cnn_light(Net* const net);

void setup_faster_rcnn(Net* const net,
                       const char* const rpn_input_name,
                       const char* const rcnn_input_name,
                       const int rpn_channels,
                       const int rpn_kernel_h, const int rpn_kernel_w,
                       const int fc_compress,
                       const int fc6_channels, const int fc7_channels,
                       const int pre_nms_topn, const int post_nms_topn);



// --------------------------------------------------------------------------
// PVANET outer interface
//   callable from external environments via shared library
// --------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

void construct_pvanet(Net* const net,
                      const char* const param_path,
                      const int is_light_model,
                      const int fc_compress,
                      const int pre_nms_topn,
                      const int post_nms_topn,
                      const int input_scale);

void set_input_pvanet(Net* const net,
                      const unsigned char* const images_data[],
                      const int heights[],
                      const int widths[],
                      const int num_images);

void get_output_pvanet(Net* const net,
                       const int image_start_index,
                       FILE* fp);

void process_pvanet(Net* const net,
                    const unsigned char image_data[],
                    const int height, const int width,
                    FILE* fp);

void process_batch_pvanet(Net* const net,
                          const unsigned char* const images_data[],
                          const int heights[],
                          const int widths[],
                          const int num_images,
                          FILE* fp);

#ifdef __cplusplus
} // end extern "C"
#endif


// --------------------------------------------------------------------------
// transform image into network input
//   img2input
// --------------------------------------------------------------------------

void img2input(const unsigned char img[],
               Tensor* const input3d,
               Tensor* const img_info1d,
               unsigned char temp_data[],
               const int height, const int width,
               const int input_scale);



// --------------------------------------------------------------------------
// shared library interface
// --------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

int _batch_size_net(void);
int _max_ndim(void);
int _max_name_len(void);

int _max_num_bottoms(void);
int _max_num_tops(void);
int _max_num_params(void);

int _max_num_tensors(void);
int _max_num_layers(void);
int _max_num_shared_blocks(void);

Net* _net(void);

void _init_net(void);
void _set_net_param_path(const char* const param_path);

void _shape_net(void);
void _malloc_net(void);

void _release_net(void);

void _detect_net(const unsigned char image_data[],
                 const int width, const int height);

Tensor* _layer_net(const int layer_id, const int top_id);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end PVA_DL_PVANET_H
