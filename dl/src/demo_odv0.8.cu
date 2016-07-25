#include "outer/pvanet.h"
#include "util/profile.h"
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#define COMPRESS

static
const char* gs_class_names[] = {
  "__unknown__",
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
  "cake_choco",
  "cake_purple",
  "cake_white",
  "sportsbag"
};

static
void draw_boxes(cv::Mat* const image,
                const real out_data[],
                const int num_boxes,
                const double time)
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
    if (time < 1000) {
      sprintf(label, "%.3lf usec", time);
    }
    else if (time < 1000000.0) {
      sprintf(label, "%.3lf msec", time / 1000);
    }
    else {
      sprintf(label, "%.3lf sec", time / 1000000.0);
    }
    cv::putText(*image, label, cv::Point(10, 10),
                2, 0.5, cv::Scalar(0, 0, 0), 2);
    cv::putText(*image, label, cv::Point(10, 10),
                2, 0.5, cv::Scalar(255, 255, 255), 1);
  }
}

static
void detect_frame(Net* const net,
                 cv::Mat* const image,
                 double* const p_mean_time,
                 int display)
{
  if (image && image->data) {
    real* boxes;
    int num_boxes;
    long int timestamp_start = tic();

    process_pvanet(net, image->data, image->rows, image->cols,
                   &boxes, &num_boxes, NULL);

    update_mean_time(p_mean_time, toc(timestamp_start));

    if (display) {
      draw_boxes(image, boxes, num_boxes, *p_mean_time);
      cv::imshow("faster-rcnn", *image);
    }
  }
}

static
void test_stream(Net* const net, cv::VideoCapture& vc)
{
  cv::Mat image;
  double mean_time = 0, sum_time = 10;

  while (1) {
    vc >> image;
    if (image.empty()) break;

    if (sum_time > 1) {
      detect_frame(net, &image, &mean_time, 1);
      sum_time = mean_time;
      if (cv::waitKey(1) == 27) { // ESC
        break;
      }
    }
    else {
      detect_frame(net, &image, &mean_time, 0);
      sum_time += mean_time;
    }
  }
}

static
void test_image(Net* const net, const char* const filename)
{
  cv::Mat image = cv::imread(filename);
  double mean_time = 0;

  if (!image.data) {
    printf("Cannot open image: %s\n", filename);
    return;
  }

  detect_frame(net, &image, &mean_time, 1);

  cv::waitKey(0);
}

static
void test_database(Net* const net,
                   const char* const db_filename)
{
  std::vector<cv::Mat> images;
  unsigned char* p_images[BATCH_SIZE];
  int image_heights[BATCH_SIZE];
  int image_widths[BATCH_SIZE];

  real* output_boxes;
  int num_output_boxes[BATCH_SIZE];

  char buf[10240];
  char* line[20];
  int count = 0, buf_count = 0;

  FILE* fp_list = fopen(db_filename, "r");
  FILE* fp_out = NULL;

  long int timestamp_start, current_time;
  double mean_time = 0;

  if (!fp_list) {
    printf("File not found: %s\n", db_filename);
  }

  while (fgets(&buf[buf_count], 1024, fp_list))
  {
    {
      const int len = strlen(&buf[buf_count]);

      buf[buf_count + len - 1] = 0;
      line[count] = &buf[buf_count];
      ++count;
      buf_count += len;
    }

    if (count == BATCH_SIZE)
    {
      for (int n = 0; n < count; ++n) {
        images.push_back(cv::imread(line[n]));
      }
      for (int n = 0; n < count; ++n) {
        p_images[n] = images[n].data;
        image_heights[n] = images[n].rows;
        image_widths[n] = images[n].cols;
      }
      for (int n = 0; n < count; ++n) {
        printf("Image %d: %s, size = %d x %d\n",
               n, line[n], image_widths[n], image_heights[n]);
      }

      timestamp_start = tic();

      process_batch_pvanet(net, p_images, image_heights, image_widths,
                           count, &output_boxes, num_output_boxes, fp_out);

      current_time = toc(timestamp_start);
      update_mean_time(&mean_time, current_time / count);
      printf("Running time: %.2lf (current), %.2lf (average)\n",
             (double)current_time / count, mean_time);

      images.clear();
      count = 0;
      buf_count = 0;
    }
  }

  if (count > 0) {
    for (int n = 0; n < count; ++n) {
      images.push_back(cv::imread(line[n]).clone());
    }
    for (int n = 0; n < count; ++n) {
      p_images[n] = images[n].data;
      image_heights[n] = images[n].rows;
      image_widths[n] = images[n].cols;
    }
    for (int n = 0; n < count; ++n) {
      printf("Image %d: %s, size = %d x %d\n",
             n, line[n], image_widths[n], image_heights[n]);
    }

    process_batch_pvanet(net, p_images, image_heights, image_widths,
                         count, &output_boxes, num_output_boxes, fp_out);
    images.clear();

    current_time = toc(timestamp_start);
    update_mean_time(&mean_time, current_time / count);
    printf("Running time: %.2lf (current), %.2lf (average)\n",
           (double)current_time / count, mean_time);
  }

  if (fp_list) {
    fclose(fp_list);
  }
  if (fp_out) {
    fclose(fp_out);
  }

  printf("Average elapsed time (microseconds)\n");
  for (int i = 0; i < net->num_layers; ++i) {
    Layer* layer = get_layer(net, i);
    printf("%s %.2lf %p\n",
           layer->name, net->elapsed_times[i], layer->f_forward);
  }
}

static
void print_usage(void)
{
  #ifdef LIGHT_MODEL
  const char* const name = "ODv0.8lite";
  #else
  const char* const name = "ODv0.8";
  #endif

  printf("[Usage] %s <command> <arg1> <arg2> ...\n\n", name);
  printf("  1. [Live demo using WebCam] %s live <camera id> <width> <height>\n", name);
  printf("  2. [Image file] %s snapshot <image filename>\n", name);
  printf("  3. [Video file] %s video <video filename>\n", name);
  printf("  4. [List of images] %s database <DB filename>\n", name);
}

static
int test(const char* const args[], const int num_args)
{
  Net* pvanet;
  const int pre_nms_topn = 6000;
  const int post_nms_topn = 200;
  const int input_size = 576;
  const char* const command = args[0];

  #ifdef GPU
  cudaSetDevice(0);
  #endif

  #ifdef LIGHT_MODEL
  const int is_light_model = 1;
  const char* const model_path = "data/pvanet_light";
  #else
  const int is_light_model = 0;
  const char* const model_path = "data/pvanet";
  #endif

  #ifdef COMPRESS
  const int fc_compress = 1;
  #else
  const int fc_compress = 0;
  #endif

  if (strcmp(command, "live") == 0) {
    if (num_args >= 4) {
      const int camera_id = atoi(args[1]);
      const int frame_width = atoi(args[2]);
      const int frame_height = atoi(args[3]);

      cv::imshow("faster-rcnn", 0);
      cv::VideoCapture vc(camera_id);
      if (!vc.isOpened()) {
        printf("Cannot open camera(%d)\n", camera_id);
        cv::destroyAllWindows();
        return -1;
      }
      vc.set(CV_CAP_PROP_FRAME_WIDTH, frame_width);
      vc.set(CV_CAP_PROP_FRAME_HEIGHT, frame_height);

      pvanet = create_pvanet(model_path, is_light_model, fc_compress,
                             pre_nms_topn, post_nms_topn, input_size);
      test_stream(pvanet, vc);
      free_net(pvanet);

      cv::destroyAllWindows();
    }
    else {
      print_usage();
      return -1;
    }
  }

  else if (strcmp(command, "snapshot") == 0) {
    if (num_args >= 2) {
      const char* const filename = args[1];
      cv::imshow("faster-rcnn", 0);

      pvanet = create_pvanet(model_path, is_light_model, fc_compress,
                             pre_nms_topn, post_nms_topn, input_size);
      test_image(pvanet, filename);
      free_net(pvanet);

      cv::destroyAllWindows();
    }
    else {
      print_usage();
      return -1;
    }
  }

  else if (strcmp(command, "video") == 0) {
    if (num_args >= 2) {
      const char* const filename = args[1];

      cv::imshow("faster-rcnn", 0);
      cv::VideoCapture vc(filename);
      if (!vc.isOpened()) {
        printf("Cannot open video: %s\n", filename);
        cv::destroyAllWindows();
        return -1;
      }

      pvanet = create_pvanet(model_path, is_light_model, fc_compress,
                             pre_nms_topn, post_nms_topn, input_size);
      test_stream(pvanet, vc);
      free_net(pvanet);

      cv::destroyAllWindows();
    }
    else {
      print_usage();
      return -1;
    }
  }

  else if (strcmp(command, "database") == 0) {
    if (num_args >= 2) {
      const char* const db_filename = args[1];

      pvanet = create_pvanet(model_path, is_light_model, fc_compress,
                             pre_nms_topn, post_nms_topn, input_size);
      test_database(pvanet, db_filename);
      free_net(pvanet);
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

  return 0;
}

int main(int argc, char* argv[])
{
  if (argc >= 3) {
    test(argv + 1, argc - 1);
  }
  else {
    print_usage();
  }

  return 0;
}
