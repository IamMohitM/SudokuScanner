#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

using contourType = std::vector<cv::Point>;
using contourVector = std::vector<contourType>;
using lineType = std::vector<cv::Vec4i>;

void drawLines(cv::Mat& houghImage, const lineType& houghLines){
	for(const auto& l : houghLines){
		line(houghImage, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]),
		     cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
	}
}

void printContourPoint(const contourType& contour){
    for(const auto& point: contour){
        std::cout << point << ",\n";
    }
}

std::tuple<cv::Point, cv::Point, cv::Point, cv::Point> findDiagonalPoints(const contourType& contour){
    int minXCoord, maxXCoord, minYCoord, maxYCoord;

    cv::Point topLeft;
    cv::Point topRight;
    cv::Point bottomLeft;
    cv::Point bottomRight;


    auto [minX, maxX] = std::minmax_element(contour.begin(), contour.end(),[](const cv::Point& a, const cv::Point& b){
        return a.x < b.x;
    });


    auto [minY, maxY] = std::minmax_element(contour.begin(), contour.end(),[](const cv::Point& a, const cv::Point& b){
        return a.y < b.y;
    });

    minXCoord = (*minX).x;
    maxXCoord = (*maxX).x;
    minYCoord = (*minY).y;
    maxYCoord = (*maxY).y;

    int minYPoint{maxYCoord}, maxYPoint{minYPoint}, minXPoint{maxXCoord}, maxXPoint{minXPoint};
    for(const auto& point: contour){
        if(point.x==minXCoord){
            if(point.y<=minYPoint){
                minYPoint = point.y;
                topLeft = point;
            }
        }

        if(point.x==maxXCoord){
            if(point.y>=maxYPoint){
                maxYPoint = point.y;
                bottomRight = point;
            }
        }

        if(point.y==minYCoord){
            if(point.x>=maxXPoint){
                maxXPoint = point.x;
                topRight = point;
            }
        }

        if(point.y==maxYCoord){
            if(point.x>=minXPoint){
                minXPoint = point.x;
                bottomLeft = point;
            }
        }
    }

    return std::make_tuple(topLeft, topRight, bottomRight, bottomLeft);
}

int main() {
	cv::Mat frame, grayImage, cannyImage, cannyHoughLineImage, cannyHoughLineImageGray, subFrame, sudokuIm, sudokuCannyImage, sudokuCannyHoughLineImage;
	contourVector contoursCanny, approxContourVector;
	cv::Size resizeDimensions{700, 512};
    cv::Size sudokuSize{400, 400};
	lineType cannyLines, cannyHierarchy, sudokuCannyLines;
	cv::Mat lambda (2, 4, CV_32FC1);
	contourType maxCannyContour, maxConvexHull, approxContour;
	cv::Point2f inputQuad[4];
	cv::Point extLeft, extRight, extTop, extBot;
	cv::Point2f outputQuad[4]  = {cv::Point2f(0, 0), cv::Point2f(sudokuSize.width -1, 0), cv::Point2f(sudokuSize.height-1,  sudokuSize.width-1),
                                  cv::Point2f(0, sudokuSize.height - 1 )} ;
	std::string cannyEdgeWindow{"Canny Frame"}, contourCannyWindow{"Contour Frame"}, cannyHoughEdgeWindow{"Canny + Hough"};
	cv::VideoCapture cap{0};
	cv::Rect maxContourRect;
//	cv::RotatedRect rotatedRect;
//	cv::Point2f rectPoints[4];
	int cannyContourIndex, key;
    double aspectRatio;
    
	while (true) {
	    sudokuIm = cv::Mat::zeros(sudokuSize, grayImage.type());
	    approxContourVector.clear();
		cap >> frame;
		if (frame.empty()) {
			std::cout << "Something went wrong while capturing the frame]=\n";
			return -1;
		}
		cv::resize(frame, frame, cv::Size(resizeDimensions));
		cv::cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);
		cv::GaussianBlur(grayImage, grayImage, cv::Size(5, 5), 1, 1);

		cv::Canny(grayImage, cannyImage, 50, 200, 3, true);
		//To thin the lines, so that surroundings are not included with the sudoku grid
		cv::erode(cannyImage, cannyImage, cv::MORPH_ERODE);
		cv::HoughLinesP(cannyImage, cannyLines, 1, CV_PI / 180, 50, 30, 1);

		cv::cvtColor(cannyImage, cannyHoughLineImage, cv::COLOR_GRAY2BGR);
		drawLines(cannyHoughLineImage, cannyLines);
		cv::cvtColor(cannyHoughLineImage, cannyHoughLineImageGray, cv::COLOR_BGR2GRAY);
		cv::findContours(cannyHoughLineImageGray, contoursCanny, cannyHierarchy, cv::RETR_TREE,
                   cv::CHAIN_APPROX_NONE);
		
		//Find the max area contour's index
		cannyContourIndex = std::distance(contoursCanny.begin(),
		                                  std::max_element(contoursCanny.begin(), contoursCanny.end(),
		                                                   [](const contourType &c1, const contourType &c2) {
			                                                   return cv::contourArea(c1) < cv::contourArea(c2);
		                                                   }));

        if (!contoursCanny.empty()) {
                maxCannyContour = contoursCanny[cannyContourIndex];
                maxContourRect = cv::boundingRect(maxCannyContour);
                aspectRatio = (maxContourRect.width*1.0)/maxContourRect.height;

            if ((1.1 > aspectRatio && aspectRatio > 0.75) &&
                    cv::contourArea(maxCannyContour) >= (0.1 * frame.rows * frame.cols)) {

                cv::approxPolyDP(maxCannyContour, approxContour, 3.0, true);
                approxContourVector.push_back(approxContour);

                extLeft  = *min_element(approxContour.begin(), approxContour.end(),
                                              [](const cv::Point& lhs, const cv::Point& rhs) {
                                                  return lhs.x < rhs.x;
                                              });
                extRight = *max_element(approxContour.begin(), approxContour.end(),
                                              [](const cv::Point& lhs, const cv::Point& rhs) {
                                                  return lhs.x < rhs.x;
                                              });
                extTop   = *min_element(approxContour.begin(), approxContour.end(),
                                              [](const cv::Point& lhs, const cv::Point& rhs) {
                                                  return lhs.y < rhs.y;
                                              });
                extBot   = *max_element(approxContour.begin(), approxContour.end(),
                                              [](const cv::Point& lhs, const cv::Point& rhs) {
                                                  return lhs.y < rhs.y;
                                              });

                inputQuad[0] = extLeft-cv::Point(5, 0);
                inputQuad[1] = extTop - cv::Point(0, 5);
                inputQuad[2] = extRight + cv::Point(5, 0);
                inputQuad[3] = extBot + cv::Point(0, 5);
                lambda = cv::getPerspectiveTransform(inputQuad, outputQuad);


//                cv::Canny(sudokuIm, sudokuCannyImage, 50, 200, 3, true);
//                //To thin the lines, so that surroundings are not included with the sudoku grid
//                cv::dilate(sudokuCannyImage, sudokuCannyImage, cv::MORPH_DILATE);
//                cv::HoughLinesP(sudokuCannyImage, sudokuCannyLines, 1, CV_PI / 180, 50, 30, 1);
//
//                cv::cvtColor(sudokuCannyImage, sudokuCannyHoughLineImage, cv::COLOR_GRAY2BGR);
//                drawLines(sudokuCannyHoughLineImage, cannyLines);
//                cv::cvtColor(sudokuCannyHoughLineImage, cannyHoughLineImageGray, cv::COLOR_BGR2GRAY);
//                cv::findContours(cannyHoughLineImageGray, contoursCanny, cannyHierarchy, cv::RETR_TREE,
//                                 cv::CHAIN_APPROX_NONE);
//                cv::drawContours(sudokuIm, contoursCanny, -1, cv::Scalar(255, 255, 255),
//                                 3, cv::LINE_8);
                cv::warpPerspective(grayImage, sudokuIm, lambda, sudokuSize);

                cv::circle(frame, extLeft, 10, cv::Scalar(255, 0, 0), 3);
                cv::circle(frame, extRight, 10, cv::Scalar(0, 255, 0), 3);
                cv::circle(frame, extTop, 10, cv::Scalar(0, 0, 255), 3);
                cv::circle(frame, extBot, 10, cv::Scalar(255, 0, 255), 3);

                cv::drawContours(frame, approxContourVector, 0, cv::Scalar(255, 255, 255),
                                                 3, cv::LINE_8, cannyHierarchy, 0);



                cv::Canny(sudokuIm, sudokuCannyImage, 50, 200, 3, true);
                //To thin the lines, so that surroundings are not included with the sudoku grid
                cv::dilate(sudokuCannyImage, sudokuCannyImage, cv::MORPH_DILATE);
                cv::HoughLinesP(sudokuCannyImage, sudokuCannyLines, 1, CV_PI / 180, 50, 30, 1);

                cv::cvtColor(sudokuCannyImage, sudokuCannyHoughLineImage, cv::COLOR_GRAY2BGR);
                drawLines(sudokuCannyHoughLineImage, cannyLines);
                cv::cvtColor(sudokuCannyHoughLineImage, cannyHoughLineImageGray, cv::COLOR_BGR2GRAY);
                cv::findContours(cannyHoughLineImageGray, contoursCanny, cannyHierarchy, cv::RETR_TREE,
                                 cv::CHAIN_APPROX_NONE);

//                auto [extLeft, extRight] = std::minmax_element(approxContour.begin(), approxContour.end(),[](const cv::Point& a, const cv::Point& b){
//                    return a.x < b.x;
//                });
//                auto [extTop, extBot] = std::minmax_element(approxContour.begin(), approxContour.end(),[](const cv::Point& a, const cv::Point& b){
//                    return a.y < b.y;
//                });
//
//                rotatedRect = cv::minAreaRect(maxCannyContour);
//                rotatedRect.points(rectPoints);
//                //draw rotated rect
//                for( int j = 0; j < 4; j++ ) {
//                    cv::line(frame, rectPoints[j], rectPoints[(j + 1) % 4],
//                             cv::Scalar(0, 0, 255), 3, 8);
//                }
            } else {
                cv::putText(frame, "Bring the Sudoku Grid Closer", cv::Point(10, 30),
                            cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 100, 0), 3);
                }

        } else {
            cv::putText(frame, "No Contours", cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX,
                        1.0, CV_RGB(255, 100, 0), 3);
        }

        cv::imshow("Sudoku", sudokuIm);
		cv::imshow(contourCannyWindow, frame);
		key = cv::waitKey(30);
        if (key == 'q') {
            break;
        }

        if(key=='a'){
            printContourPoint(approxContour);
            std::cout << '\n';
            std::cout << approxContour.size() << '\n';
            std::cout << extLeft << ",\n";
            std::cout << extTop << ",\n";
            std::cout << extRight << ",\n";
            std::cout << extBot << ",\n";
            int key1 = cv::waitKey(0);
            if(key1=='s'){
                cv::imwrite("/Users/mo/Projects/SudokuScanner/debugDirectory/frame.jpg" , frame);
            }
            if(key1=='q'){
                break;
            }
            else{
                continue;
            }
        }
    
    }

	return 0;
}

//Todo - find corners with a sliding function
//