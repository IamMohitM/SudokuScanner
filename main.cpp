#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "utils.h"


int main() {
    std::cout << "Started\n";
	cv::Mat frame, grayImage, sudokuIm, processingImage,sudokuProcessingImage;
	contourVector contoursCanny, approxContourVector;
	cv::Size resizeDimensions{700, 512}, sudokuSize{504, 504};
	lineType cannyHierarchy, sudokuCannyHierarchy;
	cv::Mat lambda (2, 4, CV_32FC1);
	contourType maxCannyContour, approxContour;
	cv::Point2f inputQuad[4];
	cv::Point2f outputQuad[4]  = {cv::Point2f(0, 0), cv::Point2f(sudokuSize.width -1, 0), cv::Point2f(sudokuSize.height-1,  sudokuSize.width-1),
                                  cv::Point2f(0, sudokuSize.height - 1 )} ;
	cv::VideoCapture cap{0};
	std::cout << "Code Block?\n";
	int maxContourIndex, key;

    std::cout << "Starting\n";
	while (true) {
	    sudokuIm = cv::Mat::zeros(sudokuSize, frame.type());
	    approxContourVector.clear();
		cap >> frame;

		if (frame.empty()) {
			std::cout << "Something went wrong while capturing the frame\n";
			return -1;
		}

		cv::resize(frame, frame, cv::Size(resizeDimensions));
		cv::cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);

		detectEdges(grayImage, processingImage);


		cv::imshow("p", processingImage);
		//To thin the lines, so that surroundings are separated from the sudoku grid
		cv::erode(processingImage, processingImage, cv::MORPH_ERODE);

		drawHoughLines(processingImage);

        cv::findContours(processingImage, contoursCanny, cannyHierarchy, cv::RETR_TREE,
                         cv::CHAIN_APPROX_NONE);

        if (!contoursCanny.empty()) {
		        //Find the max area contour's index
                maxContourIndex = getMaxContourIndex(contoursCanny);
                maxCannyContour = contoursCanny[maxContourIndex];

                if(cv::contourArea(maxCannyContour) >= (0.1 * frame.rows * frame.cols)) {
                    cv::approxPolyDP(maxCannyContour, approxContour, 3.0, true);
                    approxContourVector.push_back(approxContour);

                    auto [extLeft, extRight, extTop, extBot] = getExtremePoints(approxContour);

                    inputQuad[0] = extLeft;//  -cv::Point(5, 0);
                    inputQuad[1] = extTop;// - cv::Point(0, 5);
                    inputQuad[2] = extRight;// + cv::Point(5, 0);
                    inputQuad[3] = extBot;// + cv::Point(0, 5);

                    lambda = cv::getPerspectiveTransform(inputQuad, outputQuad);
                    cv::warpPerspective(frame, sudokuIm, lambda, sudokuSize);

                    divideSudokuGrid(sudokuIm);

                    drawCirclePoints(frame, inputQuad, 4);

                    cv::drawContours(frame, approxContourVector, 0, cv::Scalar(255, 255, 255),
                                                     3, cv::LINE_8, cannyHierarchy, 0);

            } else {
                cv::putText(frame, "Bring the Sudoku Grid Closer", cv::Point(10, 30),
                            cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 100, 0), 3);
                }

        } else {
            cv::putText(frame, "No Contours", cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX,
                        1.0, CV_RGB(255, 100, 0), 3);
        }

        cv::imshow("Sudoku", sudokuIm);
		cv::imshow("Output Frame", frame);
		key = cv::waitKey(30);
        if (key == 'q') {
            break;
        }

        if(key=='a'){
            printContourPoint(approxContour);
            std::cout << '\n';
            std::cout << approxContour.size() << '\n';
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