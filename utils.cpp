//
// Created by Mohit Motwani on 31/08/20.
//

#include "utils.h"

void drawLines(cv::Mat& houghImage, const lineType& houghLines){
    for(const auto& l : houghLines){
        cv::line(houghImage, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]),
                 cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
    }
}

void printContourPoint(const contourType& contour){
    for(const auto& point: contour){
        std::cout << point << ",\n";
    }
}

void drawConvexContours(cv::Mat& image, const contourVector& contours){
    cv::RNG rng(42);
    int index=0;
    for(const auto& contour: contours){
        if(cv::isContourConvex(contour)){
            cv::drawContours(image, contours, index, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)),
                             3, cv::LINE_8);
        }
        ++index;
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

void detectEdges(cv::Mat& inputImage, cv::Mat& processedImage, const cv::Size& filterSize,
                 int cannyMinThresh, int cannyMaxThresh, int cannyAperture, bool blur, bool debug){
    if(blur){
        cv::GaussianBlur(inputImage, processedImage, filterSize, 1, 1);
    }
    cv::Canny(processedImage, processedImage, cannyMinThresh, cannyMaxThresh, cannyAperture, true);
}

void drawHoughLines(cv::Mat& inputImage, int houghThresh, int houghMinLength, bool debug){
    lineType edgeLines;
    cv::HoughLinesP(inputImage, edgeLines, 1, CV_PI/180, houghThresh, houghMinLength, 1);
    drawLines(inputImage, edgeLines);
}

int getMaxContourIndex(const contourVector& contours){
    return std::distance(contours.begin(), std::max_element(contours.begin(), contours.end(),
                                                            [](const contourType &c1, const contourType &c2) {
                                                                    return cv::contourArea(c1) < cv::contourArea(c2);
                                                              }));
}


std::tuple<cv::Point, cv::Point, cv::Point, cv::Point> getExtremePoints(const contourType& contour){
    cv::Point extLeft  = *min_element(contour.begin(), contour.end(),
                                      [](const cv::Point& lhs, const cv::Point& rhs) {
                                          return lhs.x < rhs.x;
                                      });
    cv::Point extRight = *max_element(contour.begin(), contour.end(),
                                      [](const cv::Point& lhs, const cv::Point& rhs) {
                                          return lhs.x < rhs.x;
                                      });
    cv::Point extTop   = *min_element(contour.begin(), contour.end(),
                                      [](const cv::Point& lhs, const cv::Point& rhs) {
                                          return lhs.y < rhs.y;
                                      });
    cv::Point extBot   = *max_element(contour.begin(), contour.end(),
                                      [](const cv::Point& lhs, const cv::Point& rhs) {
                                          return lhs.y < rhs.y;
                                      });

    return std::make_tuple(extLeft, extRight, extTop, extBot);
}

void divideSudokuGrid(const cv::Mat& sudokuGrid){
    int cellWidth = sudokuGrid.cols/9;
    int cellHeight = sudokuGrid.rows/9;
    for(int i=0;i < sudokuGrid.cols - cellWidth+1; i+=cellWidth){
        for(int j=0; j<sudokuGrid.rows- cellHeight+1; j+=cellHeight){
            cv::Mat cellImg = sudokuGrid(cv::Rect(i, j, cellWidth, cellHeight));
            cv::imshow( std::to_string(i+1) + ", " + std::to_string(j+1), cellImg);
        }
    }
}

void drawCirclePoints(cv::Mat& image, cv::Point2f* points, int totalPoints){
    for(int i{0}; i< totalPoints; ++i){
        cv::circle(image, points[i], 10, cv::Scalar(255, 0, 0), 3);
    }
}