#include "pvanet.hpp"
#include <stdio.h>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <time.h>

using std::ostringstream;

using namespace cv;
using namespace std;

void draw_boxes(Mat& image, vector<pair<string, vector<float> > >& boxes, float time) {
	char text[128];
	for (int i = 0; i < (int)boxes.size(); i++) {
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
	sprintf(text, "%.3f sec", time);
	cv::putText(image, text, cv::Point(10, 10), 2, 0.5, cv::Scalar(0, 0, 0), 2);
	cv::putText(image, text, cv::Point(10, 10), 2, 0.5, cv::Scalar(255, 255, 255), 1);
}

int main(int argc, char** argv) {
	pvanet_init("", "./scripts/params3", 0);
	cv::VideoCapture vc(0);
	vc.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
	vc.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);

	cv::Mat image;
	vector<pair<string, vector<float> > > boxes;

	float time = 0;
	while (1) {
		clock_t tick0 = clock();

		vc >> image;
		if (image.empty()) break;

		pvanet_detect(image.data, image.cols, image.rows, image.step.p[0], boxes);

		clock_t tick1 = clock();
		time = (float)(tick1 - tick0) / CLOCKS_PER_SEC;

		draw_boxes(image, boxes, time);
		imshow("test_pvanet", image);

		if (cv::waitKey(1) == 27) break; //ESC
	}

	pvanet_release();
	return 0;
}
