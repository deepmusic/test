#include "pvanet.hpp"
#include <stdio.h>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "logger.h"

#ifdef THREADED
#include <omp.h>
#endif

using namespace cv;
using namespace std;

void draw_boxes(Mat& image, vector<pair<string, vector<float> > >& boxes) {
	for (int i = 0; i < (int)boxes.size(); i++) {
		char text[128];
		sprintf(text, "%s(%.2f)", boxes[i].first.c_str(), boxes[i].second[4]);
		if (boxes[i].second[4] >= 0.8) {
			rectangle(image, Rect(round(boxes[i].second[0]), round(boxes[i].second[1]), round(boxes[i].second[2] - boxes[i].second[0] + 1), round(boxes[i].second[3] - boxes[i].second[1] + 1)), Scalar(0, 0, 255), 2);
		}
		else {
			rectangle(image, Rect(round(boxes[i].second[0]), round(boxes[i].second[1]), round(boxes[i].second[2] - boxes[i].second[0] + 1), round(boxes[i].second[3] - boxes[i].second[1] + 1)), Scalar(255, 0, 0), 1);
		}
		putText(image, text, Point(round(boxes[i].second[0]), round(boxes[i].second[1] + 15)), 2, 0.5, cv::Scalar(0, 0, 0), 2);
		putText(image, text, Point(round(boxes[i].second[0]), round(boxes[i].second[1] + 15)), 2, 0.5, cv::Scalar(255, 255, 255), 1);
	}
}

int main(int argc, char** argv) {
	pvanet_init("", "./scripts/params2", 0);

    Mat image;
    if (argc >= 2)
        image = imread(argv[1]);
    else
        image = imread("./scripts/voc/004545.jpg");
        //image = imread("./test_dogs.jpg");
        //image = imread("./test_pedestrian_blur.jpg");

    vector<pair<string, vector<float> > > boxes;

    global_logger.set_log_item(PVANET_DETECT,               "pvanet_detect");
    global_logger.set_log_item(IMG2INPUT,                   "  img2input");
    global_logger.set_log_item(FORWARD_NET_INCEPTION,       "  forward_net_inception");
    global_logger.set_log_item(FORWARD_NET_RCNN,            "  forward_net_rcnn");
    global_logger.set_log_item(FORWARD_CONV_LAYER,          "    forward_conv_layer");
    global_logger.set_log_item(FORWARD_INPLACE_RELU_LAYER,  "    forward_inplace_relu_layer");
    global_logger.set_log_item(FORWARD_POOL_LAYER,          "    forward_pool_layer");
    global_logger.set_log_item(FORWARD_DECONV_LAYER,        "    forward_deconv_layer");
    global_logger.set_log_item(FORWARD_CONCAT_LAYER,        "    forward_concat_layer");
    global_logger.set_log_item(FORWARD_PROPOSAL_LAYER,      "    forward_proposal_layer");
    global_logger.set_log_item(FORWARD_ROIPOOL_LAYER,       "    forward_roipool_layer");
    global_logger.set_log_item(FORWARD_FC_LAYER,            "    forward_fc_layer");

#ifdef THREADED
    omp_set_num_threads(2);
#endif

    int iter = 10;
    if (argc >= 3)
        iter = atoi(argv[2]);

    for (int i = 0; i < iter; i++) {
        pvanet_detect(image.data, image.cols, image.rows, image.step.p[0], boxes);
        printf("iteration %d has been completed.\n", i);
    }

    global_logger.print_log();

    draw_boxes(image, boxes);
	imshow("test_pvanet", image);
	waitKey(0);

    pvanet_release();

    return 0;
}
