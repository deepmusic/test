#include "layer.h"

#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <gflags/gflags.h>

#include "boost/date_time/posix_time/posix_time.hpp"

#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_
using std::ostringstream;

DEFINE_string(cam, "0", "camera index");
DEFINE_string(width, "", "camera input width");
DEFINE_string(height, "", "camera input height");
DEFINE_string(img, "../data/voc/2007/VOC2007/JPEGImages/000004.jpg", "image path");
DEFINE_string(vid, "", "video path");
DEFINE_string(db, "", "db path");

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
    //cv::putText(*image, label, cv::Point(x1, y1 + 15),
    //        2, 0.5, cv::Scalar(0, 0, 0), 2);
    //cv::putText(*image, label, cv::Point(x1, y1 + 15),
    //        2, 0.5, cv::Scalar(255, 255, 255), 1);
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
                  cv::Mat* const image,
                  const float time)
{
  const int height = image->rows;
  const int width = image->cols;
  const int stride = image->step.p[0];

  if (image->data) {
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

    draw_boxes(image, net->output_cpu_data, net->layers[40]->tops[0].shape[0][0], time);

    cv::imshow("faster-rcnn", *image);
  }
}

static
void test_stream(Net* const net, cv::VideoCapture& vc)
{
  cv::Mat image;
  clock_t tick0 = clock();
  float time = 0;

  while (1) {
    vc >> image;
    if (image.empty()) break;

    detect_frame(net, &image, time);

    {
      clock_t tick1 = clock();
      if (time == 0) {
        time = (float)(tick1 - tick0) / CLOCKS_PER_SEC;
      }
      else {
        time = time * 0.9 + (float)(tick1 - tick0) / CLOCKS_PER_SEC * 0.1;
      }
      tick0 = tick1;
    }

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
  detect_frame(net, &image, 0);
  cv::waitKey(0);
}

static
int test(const char* const command) {
  Net frcnn;

  #ifdef GPU
  cudaSetDevice(0);
  #endif

  construct_frcnn_7_1_1(&frcnn);

  if (strcmp(command, "live") == 0) {
    cv::VideoCapture vc(boost::lexical_cast<int>(FLAGS_cam));
    if (!vc.isOpened()) {
      printf("Cannot open camera(%s)\n", FLAGS_cam.c_str());
      return -1;
    }
    vc.set(CV_CAP_PROP_FRAME_WIDTH, boost::lexical_cast<int>(FLAGS_width));
    vc.set(CV_CAP_PROP_FRAME_HEIGHT, boost::lexical_cast<int>(FLAGS_height));
    test_stream(&frcnn, vc);
  }
  else if (strcmp(command, "snapshot") == 0) {
    test_image(&frcnn, FLAGS_img.c_str());
  }
  else if (strcmp(command, "video") == 0) {
    cv::VideoCapture vc(boost::lexical_cast<int>(FLAGS_vid));
    if (!vc.isOpened()) {
      printf("Cannot open video: %s\n", FLAGS_vid.c_str());
      return -1;
    }
    test_stream(&frcnn, vc);
  }
  else if (strcmp(command, "database") == 0) {
    char path[256];
    FILE* fp = fopen(FLAGS_db.c_str(), "r");
    if (!fp) {
      printf("Cannot open db: %s\n", FLAGS_db.c_str());
      return -1;
    }
    while (fgets(path, 256, fp)) {
      path[strlen(path) - 1] = 0;
      test_image(&frcnn, path);
    }
    fclose(fp);
  }

  cv::destroyAllWindows();

  return 0;
}

int main(int argc, char* argv[]) {
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
    "usage: ./demo.bin <command> <args>\n\n"
    "commands:\n"
    "  live     \n"
    "  snapshot       \n"
    "  video    \n"
    "  database       \n");
  
  if (argc == 2) {
    test(argv[1]);
  }
  else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "demo");
  }

  return 0;
}
