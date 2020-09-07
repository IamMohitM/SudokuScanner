//
// Created by Mohit Motwani on 31/08/20.
//

#pragma once
#include <opencv2/opencv.hpp>

using contourType = std::vector<cv::Point>;
using contourVector = std::vector<contourType>;
using lineType = std::vector<cv::Vec4i>;

void drawLines(cv::Mat& houghImage, const lineType& houghLines);
void printContourPoint(const contourType& contour);
void drawConvexContours(cv::Mat& image, const contourVector& contours);
std::tuple<cv::Point, cv::Point, cv::Point, cv::Point> findDiagonalPoints(const contourType& contour);
void detectEdges(cv::Mat& inputImage, cv::Mat& processedImage, const cv::Size& filterSize=cv::Size(5,5),
                 int cannyMinThresh=50, int cannyMaxThresh=200, int cannyAperture=3, bool blur=true);
void drawHoughLines(cv::Mat& inputImage, int houghThresh=50, int houghMinLength=40);
int getMaxContourIndex(const contourVector& contours);
std::tuple<cv::Point, cv::Point, cv::Point, cv::Point> getExtremePoints(const contourType& contour);
void divideSudokuGrid(const cv::Mat& graySudokuGrid, const cv::Mat& sudokuGrid);

void drawCirclePoints(cv::Mat& image, cv::Point2f* points, int totalPoints);
