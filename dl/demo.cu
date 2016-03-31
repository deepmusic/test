#include "layer.h"

#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#ifdef _MSC_VER
  #include <time.h>
#else
  #include "boost/date_time/posix_time/posix_time.hpp"
#endif

using std::ostringstream;

static
const char* gs_class_names[] = {
  "none",
  "bicycle",
  "bird",
  "bus",
  "car",
  "cat",
  "dog",
  "horse",
  "motorbike",
  "person",
  "train",
  "aeroplane",
  "boat",
  "bottle",
  "chair",
  "cow",
  "diningtable",
  "pottedplant",
  "sheep",
  "sofa",
  "tvmonitor",
  "cake",
  "vase"
};

static
void get_output(Net* net)
{
  const Tensor* const out = &net->layers[40]->tops[0];
  const long int output_size = flatten_size(out);

  #ifdef GPU
  cudaMemcpyAsync(net->output_cpu_data, out->data,
                  output_size * sizeof(real),
                  cudaMemcpyDeviceToHost);
  #else
  memcpy(net->output_cpu_data, out->data, output_size * sizeof(real));
  #endif
}

static
void draw_boxes(cv::Mat* const image,
                const real* const out_data,
                const int num_boxes,
                const float time)
{
  char label[128];
  for (int r = 0; r < num_boxes; ++r) {
    const real* const p_box = out_data + r * 6;
    const char* const class_name = gs_class_names[(int)p_box[0]];
    const real score = p_box[5];
    const int x1 = (int)ROUND(p_box[1]);
    const int y1 = (int)ROUND(p_box[2]);
    const int x2 = (int)ROUND(p_box[3]);
    const int y2 = (int)ROUND(p_box[4]);
    const int w = x2 - x1 + 1;
    const int h = y2 - y1 + 1;
    sprintf(label, "%s(%.2f)", class_name, score);

    if (score >= 0.8) {
      cv::rectangle(*image, cv::Rect(x1, y1, w, h),
                    cv::Scalar(0, 0, 255), 2);
    }
    else {
      cv::rectangle(*image, cv::Rect(x1, y1, w, h),
                    cv::Scalar(255, 0, 0), 1);
    }
    cv::putText(*image, label, cv::Point(x1, y1 + 15),
            2, 0.5, cv::Scalar(0, 0, 0), 2);
    cv::putText(*image, label, cv::Point(x1, y1 + 15),
            2, 0.5, cv::Scalar(255, 255, 255), 1);
  }
  if (time > 0) {
    sprintf(label, "%.3f sec", time);
    cv::putText(*image, label, cv::Point(10, 10),
                2, 0.5, cv::Scalar(0, 0, 0), 2);
    cv::putText(*image, label, cv::Point(10, 10),
                2, 0.5, cv::Scalar(255, 255, 255), 1);
  }
}

static
void detect_frame(Net* const net,
                  cv::Mat* const image)
{
  if (image && image->data) {
    const clock_t tick0 = clock();
    real time = 0;

    const int height = image->rows;
    const int width = image->cols;
    const int stride = (int)image->step.p[0];

    Tensor* input = &net->layers[0]->tops[0];
    input->ndim = 3;
    input->num_items = 0;
    input->start[0] = 0;
    net->img_info->ndim = 1;
    net->img_info->num_items = 0;

    img2input(image->data, &net->layers[0]->tops[0], net->img_info,
              (unsigned char*)net->temp_data,
              height, width, stride);

    shape_net(net);

    forward_net(net);

    get_output(net);

    {
      clock_t tick1 = clock();
      if (time == 0) {
        time = (real)(tick1 - tick0) / CLOCKS_PER_SEC;
      }
      else {
        time = time * 0.9f + (real)(tick1 - tick0) / CLOCKS_PER_SEC * 0.1f;
      }
    }

    draw_boxes(image, net->output_cpu_data,
               net->layers[40]->tops[0].shape[0][0],
               time);

    cv::imshow("faster-rcnn", *image);
  }
}

static
void test_stream(Net* const net, cv::VideoCapture& vc)
{
  cv::Mat image;

  while (1) {
    vc >> image;
    if (image.empty()) break;

    detect_frame(net, &image);

    if (cv::waitKey(1) == 27) break; //ESC
  }
}

static
void test_image(Net* const net, const char* const filename)
{
  cv::Mat image = cv::imread(filename);
  if (!image.data) {
    printf("Cannot open image: %s\n", filename);
    return;
  }

  detect_frame(net, &image);

  cv::waitKey(0);
}

static
void print_usage(void)
{
  printf("[Usage] ./demo_gpu.bin <command> <args>\n\n");
  printf("  1. [Live demo using WebCam] ./demo_gpu.bin live\n");
  printf("  2. [Image file] ./demo_gpu.bin snapshot <filename>\n");
  printf("  3. [Video file] ./demo_gpu.bin video <filename>\n");
  printf("  4. [List of images] ./demo_gpu.bin database <filename>\n");
}

static
int test(const char* const args[], const int num_args)
{
  Net frcnn;
  const char* const command = args[0];

  cv::imshow("faster-rcnn", 0);

  #ifdef GPU
  cudaSetDevice(0);
  #endif

  construct_frcnn_7_1_1(&frcnn);

  if (strcmp(command, "live") == 0) {
    cv::VideoCapture vc(0);
    if (!vc.isOpened()) {
      printf("Cannot open camera(0)\n");
      return -1;
    }
    vc.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
    vc.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
    test_stream(&frcnn, vc);
  }

  else if (strcmp(command, "snapshot") == 0) {
    if (num_args > 1) {
      test_image(&frcnn, args[1]);
    }
    else {
      print_usage();
      return -1;
    }
  }

  else if (strcmp(command, "video") == 0) {
    if (num_args > 1) {
      cv::VideoCapture vc(args[1]);
      if (!vc.isOpened()) {
        printf("Cannot open video: %s\n", args[1]);
        return -1;
      }
      test_stream(&frcnn, vc);
    }
    else {
      print_usage();
      return -1;
    }
  }

  else if (strcmp(command, "database") == 0) {
    if (num_args > 1) {
      char path[256];
      FILE* fp = fopen(args[1], "r");
      if (!fp) {
        printf("Cannot open db: %s\n", args[1]);
        return -1;
      }
      while (fgets(path, 256, fp)) {
        path[strlen(path) - 1] = 0;
        test_image(&frcnn, path);
      }
      fclose(fp);
    }
    else {
      print_usage();
      return -1;
    }
  }

  else {
    print_usage();
    return -1;
  }

  cv::destroyAllWindows();
  return 0;
}

int main(int argc, char* argv[])
{
  if (argc >= 2) {
    test(argv + 1, argc - 1);
  }
  else {
    print_usage();
  }

  return 0;
}
