#include "layer.h"
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <time.h>

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

    process_pvanet(net, image->data, image->rows, image->cols, NULL);

    {
      clock_t tick1 = clock();
      if (time == 0) {
        time = (real)(tick1 - tick0) / CLOCKS_PER_SEC;
      }
      else {
        time = time * 0.9f + (real)(tick1 - tick0) / CLOCKS_PER_SEC * 0.1f;
      }
    }

    draw_boxes(image, net->temp_cpu_data, net->num_output_boxes, time);

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
void test_database(Net* const net,
                   const char* const db_filename,
                   const char* const out_filename)
{
  #if BATCH_SIZE == 4
  static const int batch_size = 4;
  #else
  static const int batch_size = 1;
  #endif

  char buf[10240];
  char* line[20];
  int total_count = 0, count = 0, buf_count = 0;
  FILE* fp_list = fopen(db_filename, "r");

  #ifndef DEMO
  FILE* fp_out = fopen(out_filename, "wb");
  #else
  FILE* fp_out = NULL;
  #endif

  clock_t tick0, tick1;
  float a_time[2] = { 0, };

  if (!fp_list) {
    printf("File not found: %s\n", db_filename);
  }

  #ifndef DEMO
  if (!fp_out) {
    printf("File write error: %s\n", out_filename);
  }
  #endif

  tick0 = clock();

  while (fgets(&buf[buf_count], 1024, fp_list))
  {
    {
      const int len = strlen(&buf[buf_count]);

      buf[buf_count + len - 1] = 0;
      line[count] = &buf[buf_count];
      ++count;
      buf_count += len;
    }

    if (count == batch_size)
    {
    #if BATCH_SIZE == 4
      cv::Mat images[] = {
        cv::imread(line[0]), cv::imread(line[1]),
        cv::imread(line[2]), cv::imread(line[3])
      };
      const unsigned char* const images_data[] = {
        images[0].data, images[1].data, images[2].data, images[3].data
      };
      const int heights[] = {
        images[0].rows, images[1].rows, images[2].rows, images[3].rows
      };
      const int widths[] = {
        images[0].cols, images[1].cols, images[2].cols, images[3].cols
      };
    #else
      cv::Mat images[] = { cv::imread(line[0]) };
      const unsigned char* const images_data[] = { images[0].data };
      const int heights[] = { images[0].rows };
      const int widths[] = { images[0].cols };
    #endif

      process_batch_pvanet(net, images_data, heights, widths, batch_size,
                           fp_out);

      tick1 = clock();
      a_time[0] = (float)(tick1 - tick0) / CLOCKS_PER_SEC;
      a_time[1] += (float)(tick1 - tick0) / CLOCKS_PER_SEC;
      tick0 = tick1;
      printf("Running time: %.2f (current), %.2f (average)\n",
             a_time[0] * 1000 / count,
             a_time[1] * 1000 / (total_count + count));

      total_count += count;
      count = 0;
      buf_count = 0;
    }
  }

  if (count > 0) {
    for (int n = 0; n < count; ++n) {
      cv::Mat image = cv::imread(line[n]);
      process_pvanet(net, image.data, image.rows, image.cols, fp_out);
    }

    tick1 = clock();
    a_time[0] = (float)(tick1 - tick0) / CLOCKS_PER_SEC;
    a_time[1] += (float)(tick1 - tick0) / CLOCKS_PER_SEC;
    tick0 = tick1;
    printf("Running time: %.2f (current), %.2f (average)\n",
           a_time[0] * 1000 / count,
           a_time[1] * 1000 / (total_count + count));
  }

  if (fp_list) {
    fclose(fp_list);
  }
  if (fp_out) {
    fclose(fp_out);
  }
}

static
void print_usage(void)
{
  printf("[Usage] ./demo_gpu.bin <command> <model path> <arg1> <arg2> ...\n");
  printf("  1. [Live demo using WebCam] ./demo_gpu.bin live <model path> <camera id> <width> <height>\n");
  printf("  2. [Image file] ./demo_gpu.bin snapshot <model path> <image filename>\n");
  printf("  3. [Video file] ./demo_gpu.bin video <model path> <video filename>\n");
  printf("  4. [List of images] ./demo_gpu.bin database <model path> <DB filename> <output filename>\n");
}

static
int test(const char* const args[], const int num_args)
{
  Net pvanet;
  const char* const command = args[0];
  const char* const model_path = args[1];

  #ifdef GPU
  cudaSetDevice(0);
  #endif

  construct_pvanet(&pvanet, model_path);

  if (strcmp(command, "live") == 0) {
    if (num_args >= 5) {
      const int camera_id = atoi(args[2]);
      const int frame_width = atoi(args[3]);
      const int frame_height = atoi(args[4]);

      cv::imshow("faster-rcnn", 0);
      cv::VideoCapture vc(camera_id);
      if (!vc.isOpened()) {
        printf("Cannot open camera(%d)\n", camera_id);
        cv::destroyAllWindows();
        return -1;
      }
      vc.set(CV_CAP_PROP_FRAME_WIDTH, frame_width);
      vc.set(CV_CAP_PROP_FRAME_HEIGHT, frame_height);
      test_stream(&pvanet, vc);
      cv::destroyAllWindows();
    }
    else {
      print_usage();
      return -1;
    }
  }

  else if (strcmp(command, "snapshot") == 0) {
    if (num_args > 2) {
      const char* const filename = args[2];

      cv::imshow("faster-rcnn", 0);
      test_image(&pvanet, filename);
      cv::destroyAllWindows();
    }
    else {
      print_usage();
      return -1;
    }
  }

  else if (strcmp(command, "video") == 0) {
    if (num_args > 2) {
      const char* const filename = args[2];

      cv::imshow("faster-rcnn", 0);
      cv::VideoCapture vc(filename);
      if (!vc.isOpened()) {
        printf("Cannot open video: %s\n", filename);
        cv::destroyAllWindows();
        return -1;
      }
      test_stream(&pvanet, vc);
      cv::destroyAllWindows();
    }
    else {
      print_usage();
      return -1;
    }
  }

  else if (strcmp(command, "database") == 0) {
    if (num_args > 3) {
      const char* const db_filename = args[2];
      const char* const out_filename = args[3];

      test_database(&pvanet, db_filename, out_filename);
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

#ifdef TEST
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
#endif
