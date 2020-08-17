#include <iostream>
#include <opencv2/opencv.hpp>

using contourVector = std::vector<std::vector<cv::Point>>;
using contourType = std::vector<cv::Point>;
using lineType = std::vector<cv::Vec4i>;

void drawLines(cv::Mat& houghImage, const lineType& houghLines){
	for(const auto& l : houghLines){
		line(houghImage, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]),
		     cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
	}
}

int main() {
	cv::Mat frame, grayImage, cannyImage,
			cannyHoughLineImage, cannyHoughLineImageGray;
	contourVector contoursCanny, contoursThresh;
	lineType cannyLines, cannyHierarchy;
	contourType maxCannyContour;
	std::string cannyEdgeWindow{"Canny Frame"}, contourCannyWindow{"Contour Frame"}, cannyHoughEdgeWindow{"Canny + Hough"};
	int cannyContourIndex, key;
	cv::VideoCapture cap{0};
	cv::Rect maxContourRect;
	std::string msg = "Bring the block closer";

//	contourVector outRect;
//	cv::Mat outRect;
//	frame = cv::imread("/home/mo/CLionProjects/SudokuSolver/images/test_3.jpg");
//	cv::resize(frame, frame, cv::Size( 512, 512));
	while (true) {
		cap >> frame;
		if (frame.empty()) {
			std::cout << "Something went wrong while capturing the frame]=\n";
			return -1;
		}
		cv::cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);
		cv::GaussianBlur(grayImage, grayImage, cv::Size(5, 5), 1, 1);

		cv::Canny(grayImage, cannyImage, 50, 200, 3, true);
		//To thin the lines, so that surroundings are not included with the sudoku grid
		cv::erode(cannyImage, cannyImage, cv::MORPH_ERODE);
		cv::imshow(cannyEdgeWindow, cannyImage);
		cv::HoughLinesP(cannyImage, cannyLines, 1, CV_PI / 180, 50, 30, 1);

		cv::cvtColor(cannyImage, cannyHoughLineImage, cv::COLOR_GRAY2BGR);
		drawLines(cannyHoughLineImage, cannyLines);
		cv::imshow(cannyHoughEdgeWindow, cannyHoughLineImage);
		cv::cvtColor(cannyHoughLineImage, cannyHoughLineImageGray, cv::COLOR_BGR2GRAY);
		cv::findContours(cannyHoughLineImageGray, contoursCanny, cannyHierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
		//Find the max area contour's index
		cannyContourIndex = std::distance(contoursCanny.begin(),
		                                  std::max_element(contoursCanny.begin(), contoursCanny.end(),
		                                                   [](const contourType &c1, const contourType &c2) {
			                                                   return cv::contourArea(c1) < cv::contourArea(c2);
		                                                   }));

		if(cv::contourArea(contoursCanny[cannyContourIndex]) >= 0.3                              * frame.rows * frame.cols) {
			maxContourRect = cv::boundingRect(contoursCanny[cannyContourIndex]);

			cv::drawContours(frame, contoursCanny, cannyContourIndex, cv::Scalar(200, 0, 100), -1, cv::LINE_8,
			                 cannyHierarchy, 0);
//		cv::drawContours(frame, outRect, 0, cv::Scalar( 0, 0, 100),2, cv::LINE_8, cannyHierarchy, 0);
			cv::rectangle(frame, maxContourRect, cv::Scalar(0, 255, 0));

		}else{
			std::cout << "Bring the block closer\n";
			std::cout << cv::FONT_HERSHEY_COMPLEX << '\n';
//			cv::addText(frame, "dsfasdf", cv::Point(0, 10), cv::FONT_HERSHEY_COMPLEX);//, 1.4, cv::Scalar(0, 0, 255), 2, 2, 1);
//			cv::addText(frame, "msg",cv::Point(10, frame.rows/2), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 185, 0), 10);
		}
		cv::imshow(contourCannyWindow, frame);

		key = cv::waitKey(1);
	if (key == 'q') {
		break;
	}

}

	return 0;
}
// TODO: Check with convex hull
// TODO: Check with rotated rect