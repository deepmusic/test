#include "outer/pvanet.h"
#include <string.h>

// --------------------------------------------------------------------------
// kernel code
//   set_input_pvanet: set given images to network instance
//   get_output_pvanet: get object detection results
// --------------------------------------------------------------------------

static
void set_input_pvanet(Net* const net,
                      unsigned char* const p_images[],
                      const int image_heights[],
                      const int image_widths[],
                      const int num_images)
{
  int shape_changed = (net->num_images != num_images);

  // if the number of images exceeds batch size, ignore the excess
  if (num_images > BATCH_SIZE) {
    printf("[ERROR] Number of input images %d exceeds batch size %d\n",
           num_images, BATCH_SIZE);
    printf("        Process first %d input images only\n",
           BATCH_SIZE);
    net->num_images = BATCH_SIZE;
  }
  else {
    net->num_images = num_images;
  }

  // set images as network inputs
  // with checking whether current image shapes == previous image shapes
  for (int n = 0; n < net->num_images; ++n) {
    net->p_images[n] = p_images[n];

    if (net->image_heights[n] != image_heights[n] ||
        net->image_widths[n] != image_widths[n])
    {
      shape_changed = 1;
      net->image_heights[n] = image_heights[n];
      net->image_widths[n] = image_widths[n];
    }
  }

  // if current image shapes != previous image shapes,
  // recompute output shapes for all layers
  if (shape_changed) {
    printf("shape changed\n");
    shape_net(net);
  }
}

static
void get_output_pvanet(Net* const net,
                       real** const ref_output_boxes,
                       int num_boxes[],
                       FILE* fp)
{
  const Tensor* const tensor = get_tensor_by_name(net, "out");
  real* p_output;

  #ifdef GPU
  {
    // in GPU mode, copy GPU memory -> main memory first
    const long int data_size = get_data_size(tensor);
    cudaMemcpyAsync(net->temp_cpu_data, tensor->data,
                    data_size * sizeof(real),
                    cudaMemcpyDeviceToHost);
    p_output = net->temp_cpu_data;
  }
  #else
  {
    p_output = tensor->data;
  }
  #endif

  // store reference to output data at main memory
  // and store number of output boxes in each input image
  if (ref_output_boxes) {
    *ref_output_boxes = p_output;
  }
  if (num_boxes) {
    for (int n = 0; n < tensor->num_items; ++n) {
      num_boxes[n] = tensor->shape[n][0];
    }
  }

  // save output data to binary, used for measuring accuracy
  if (fp) {
    for (int n = 0; n < tensor->num_items; ++n) {
      const real* const p_out_item = p_output + tensor->start[n];
      fwrite(&tensor->ndim, sizeof(int), 1, fp);
      fwrite(tensor->shape[n], sizeof(int), tensor->ndim, fp);
      fwrite(p_out_item, sizeof(real), tensor->shape[n][0] * 6, fp);
    }
  }

  // print output data
  for (int n = 0; n < tensor->num_items; ++n) {
    const real* const p_out_item = p_output + tensor->start[n];

    for (int i = 0; i < tensor->shape[n][0]; ++i) {
      const int predicted_class = (int)p_out_item[i * 6 + 0];
      const real score = p_out_item[i * 6 + 5];

      printf("Image %d / Box %d: ", n, i);
      printf("class %d, score %f, p1 = (%.2f, %.2f), p2 = (%.2f, %.2f)\n",
             predicted_class, score,
             p_out_item[i * 6 + 1], p_out_item[i * 6 + 2],
             p_out_item[i * 6 + 3], p_out_item[i * 6 + 4]);
    }
  }
}



// --------------------------------------------------------------------------
// PVANET interface
//   create_pvanet: construct new PVANET instance and return it
//   process_pvanet: perform object detection for given image
//   process_batch_pvanet: perform object detection for given images
// --------------------------------------------------------------------------

Net* create_pvanet(const char* const param_path,
                   const int is_light_model,
                   const int fc_compress,
                   const int pre_nms_topn,
                   const int post_nms_topn,
                   const int input_size)
{
  Net* net = create_empty_net();
  strcpy(net->param_path, param_path);

  if (is_light_model) {
    setup_shared_cnn_light(net, input_size);
    setup_faster_rcnn(net, "convf", "convf", 256, 1, 1,
        fc_compress, 512, 128, pre_nms_topn, post_nms_topn);
  }
  else {
    setup_shared_cnn(net, input_size);
    setup_faster_rcnn(net, "convf_rpn", "convf", 384, 3, 3,
        fc_compress, 512, 512, pre_nms_topn, post_nms_topn);
  }

  malloc_net(net);

  return net;
}

void process_pvanet(Net* const net,
                    unsigned char image[],
                    const int height,
                    const int width,
                    real** const ref_output_boxes,
                    int* const p_num_boxes,
                    FILE* fp)
{
  set_input_pvanet(net, &image, &height, &width, 1);

  forward_net(net);

  get_output_pvanet(net, ref_output_boxes, p_num_boxes, fp);
}

void process_batch_pvanet(Net* const net,
                          unsigned char* const p_images[],
                          const int heights[],
                          const int widths[],
                          const int num_images,
                          real** const ref_output_boxes,
                          int num_boxes[],
                          FILE* fp)
{
  set_input_pvanet(net, p_images, heights, widths, num_images);

  forward_net(net);

  get_output_pvanet(net, ref_output_boxes, num_boxes, fp);
}
