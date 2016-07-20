#include "pvanet.h"
#include <string.h>

void construct_pvanet(Net* const pvanet,
                      const char* const param_path,
                      const int is_light_model,
                      const int fc_compress,
                      const int pre_nms_topn,
                      const int post_nms_topn,
                      const int input_scale)
{
  init_net(pvanet);

  strcpy(pvanet->param_path, param_path);
  pvanet->input_scale = input_scale;

  if (is_light_model) {
    setup_shared_cnn_light(pvanet);
    setup_faster_rcnn(pvanet, "convf", "convf", 256, 1, 1,
        fc_compress, 512, 128, pre_nms_topn, post_nms_topn);
  }
  else {
    setup_shared_cnn(pvanet);
    setup_faster_rcnn(pvanet, "convf_rpn", "convf", 384, 3, 3,
        fc_compress, 512, 512, pre_nms_topn, post_nms_topn);
  }

  shape_net(pvanet);

  malloc_net(pvanet);
}

void set_input_pvanet(Net* const net,
                      const unsigned char* const images_data[],
                      const int heights[],
                      const int widths[],
                      const int num_images)
{

  Tensor* const input = get_tensor_by_name(net, "data");
  Tensor* const img_info = get_tensor_by_name(net, "im_info");
  int shape_changed = (input->num_items != num_images);

  if (!shape_changed) {
    #ifdef GPU
    real img_info_cpu[BATCH_SIZE * 6];
    const real* const p_img_info_cpu = img_info_cpu;
    cudaMemcpyAsync(img_info_cpu, img_info->data,
                    get_data_size(img_info) * sizeof(real),
                    cudaMemcpyDeviceToHost);
    #else
    const real* const p_img_info_cpu = img_info->data;
    #endif
    for (int n = 0; n < num_images; ++n) {
      if (p_img_info_cpu[n * 6 + 4] != (real)heights[n] ||
          p_img_info_cpu[n * 6 + 5] != (real)widths[n])
      {
        shape_changed = 1;
        break;
      }
    }
  }

  input->ndim = 3;
  input->num_items = 0;
  input->start[0] = 0;

  img_info->ndim = 1;
  img_info->num_items = 0;

  for (int n = 0; n < num_images; ++n) {
    img2input(images_data[n], input, img_info,
              (unsigned char*)net->temp_data,
              heights[n], widths[n], net->input_scale);
  }

  if (shape_changed) {
    printf("shape changed\n");
    shape_net(net);
  }
}

void get_output_pvanet(Net* const net,
                       const int image_start_index,
                       FILE* fp)
{
  // retrieve & save test output for measuring performance
  const Tensor* const out = get_tensor_by_name(net, "out");
  const long int output_size = get_data_size(out);

  #ifdef GPU
  cudaMemcpyAsync(net->temp_cpu_data, out->data,
                  output_size * sizeof(real),
                  cudaMemcpyDeviceToHost);
  #else
  memcpy(net->temp_cpu_data, out->data, output_size * sizeof(real));
  #endif

  if (fp) {
    for (int n = 0; n < out->num_items; ++n) {
      const real* const p_out_item = net->temp_cpu_data + out->start[n];
      fwrite(&out->ndim, sizeof(int), 1, fp);
      fwrite(out->shape[n], sizeof(int), out->ndim, fp);
      fwrite(p_out_item, sizeof(real), out->shape[n][0] * 6, fp);
    }
  }

  net->num_output_boxes = 0;
  for (int n = 0; n < out->num_items; ++n) {
    const int image_index = image_start_index + n;
    const real* const p_out_item = net->temp_cpu_data + out->start[n];

    for (int i = 0; i < out->shape[n][0]; ++i) {
      const int class_index = (int)p_out_item[i * 6 + 0];

      if (p_out_item[i * 6 + 5] < 0.7f) {
        continue;
      }
      printf("Image %d / Box %d: ", image_index, i);
      printf("class %d, score %f, p1 = (%.2f, %.2f), p2 = (%.2f, %.2f)\n",
             class_index, p_out_item[i * 6 + 5],
             p_out_item[i * 6 + 1], p_out_item[i * 6 + 2],
             p_out_item[i * 6 + 3], p_out_item[i * 6 + 4]);
    }
    net->num_output_boxes += out->shape[n][0];
  }
}

void process_pvanet(Net* const net,
                    const unsigned char image_data[],
                    const int height,
                    const int width,
                    FILE* fp)
{
  set_input_pvanet(net, &image_data, &height, &width, 1);

  forward_net(net);

  get_output_pvanet(net, 0, fp);
}

void process_batch_pvanet(Net* const net,
                          const unsigned char* const images_data[],
                          const int heights[],
                          const int widths[],
                          const int num_images,
                          FILE* fp)
{
  set_input_pvanet(net, images_data, heights, widths, num_images);

  forward_net(net);

  get_output_pvanet(net, 0, fp);
}
