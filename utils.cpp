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

void detectEdges(cv::Mat& inputImage, cv::Mat& processedImage, const cv::Size& filterSize,
                 int cannyMinThresh, int cannyMaxThresh, int cannyAperture, bool blur){
    if(blur){
        cv::GaussianBlur(inputImage, processedImage, filterSize, 1, 1);
    }
    cv::Canny(processedImage, processedImage, cannyMinThresh, cannyMaxThresh, cannyAperture, true);
}

void drawHoughLines(cv::Mat& inputImage, int houghThresh, int houghMinLength){
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

bool contaisDigit(cv::Mat& img, float thresh = 50, bool invertImage=false){
    float maxArea = -1;
    cv::Mat labelImage, stats, centroids;

    if(invertImage){
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        cv::threshold(img, img, 100, 255, cv::THRESH_BINARY_INV);
    }
    int nlabels = cv::connectedComponentsWithStats(img, labelImage, stats, centroids, 8, CV_32S);
    for(int label = 1; label<nlabels; ++label){
        if(maxArea < stats.at<int>(label, cv::CC_STAT_AREA)){
            maxArea = stats.at<int>(label, cv::CC_STAT_AREA);
        }
    }

    return maxArea >= thresh;
}

void divideSudokuGrid(const cv::Mat& graySudokuGrid, const cv::Mat& sudokuGrid){
//    cv::cvtColor(sudokuGrid, graySudokuGrid, cv::COLOR_BGR2GRAY);

    contourVector contours;
    int cellWidth = graySudokuGrid.cols / 9;
    int cellHeight = graySudokuGrid.rows / 9;
    int nlabels;
    cv::Size resize = cv::Size(96, 96);
    cv::Mat labelImage(cv::Size(cellWidth, cellHeight), CV_32S), stats, centroids;
    double maxArea = -100;
    cv::Mat morphed, cellImg, cellColor, mask(labelImage.size(), CV_8UC1, cv::Scalar(0));
    auto kernel = cv::getStructuringElement(2, cv::Size(7, 7));
    cv::Rect rect;
    std::vector<std::tuple<int, int>> cellsWithDigits{};

    for(int i=0, row=1; i < graySudokuGrid.rows; i+=cellHeight, ++row){
        for(int j=0, col=1; j < graySudokuGrid.cols; j+=cellWidth, ++col){
            cellImg = graySudokuGrid(cv::Rect(j, i, cellWidth, cellHeight));
            cellColor = sudokuGrid(cv::Rect(j, i, cellWidth, cellHeight));

            cv::morphologyEx(cellImg, morphed, cv::MORPH_CLOSE, kernel);
            cv::threshold(morphed, morphed, 128, 255, cv::THRESH_BINARY_INV);
            nlabels = cv::connectedComponentsWithStats(morphed, labelImage, stats, centroids, 8, CV_32S);
            for(int label = 1; label<nlabels; ++label){
                if(maxArea < stats.at<int>(label, cv::CC_STAT_AREA)){
                    mask = labelImage==label;
                    maxArea = stats.at<int>(label, cv::CC_STAT_AREA);
                }
            }

//            float avg = static_cast<float>(cv::mean(cv::mean(cellImg, mask)).val[0]);
            cv::Mat roi(cellImg.size(), cellImg.type(), cv::Scalar(0));
            roi.setTo(cv::Scalar(255), mask);

            cv::findContours(roi, contours, cv::noArray(), cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            rect = cv::boundingRect(contours[0]);
            cv::Mat extractedImg = cellColor(rect);

            if(extractedImg.rows > 0 && extractedImg.cols > 0) {
                if(contaisDigit(extractedImg, 60, true)){
                    cellsWithDigits.push_back(std::make_tuple(row, col));
                    cv::imshow(std::to_string(row) + ", " + std::to_string(col), extractedImg);
                    cv::moveWindow(std::to_string(row) + ", " + std::to_string(col), i + cellWidth, j + cellHeight);
                }
            }


            maxArea = -100;
            mask = cv::Scalar(0);
        }
    }
//        cv::destroyAllWindows();

//    std::cout << cellsWithDigits.size() << ": ";
//    for(const auto& pair: cellsWithDigits){
//        std::cout << "(" << std::get<0>(pair) << ", " << std::get<1>(pair) << ") ";
//    }
//    std::cout << '\n';

}

void drawCirclePoints(cv::Mat& image, cv::Point2f* points, int totalPoints){
    for(int i{0}; i< totalPoints; ++i){
        cv::circle(image, points[i], 10, cv::Scalar(255, 0, 0), 3);
    }
}
