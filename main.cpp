#include <iostream>
#include <vector>
#include "classes/shapes/circle.h"
#include "classes/chromosomes/simpleChromosome.h"

#include "opencv2/core/mat.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <chrono>

using namespace vidga;

int main() {
    ucoor_t xRes = 640, yRes = 360;
    const auto minRadius = static_cast<ucoor_t>((3.0/100) * xRes);
    const auto maxRadius = static_cast<ucoor_t>((10.0/100) * xRes);
    const float avgRadius = static_cast<ucoor_t>((minRadius + maxRadius) / 2);

    auto canvas = cv::Mat(yRes, xRes, CV_8UC3);

    const auto avgCircleSize = (avgRadius * avgRadius * 3.14159265);
    const auto circleAmountFactor = 2.5;
    auto numCircles = static_cast<size_t >(circleAmountFactor * xRes * yRes / avgCircleSize);

    for (auto i = 0; i < 4; i++) {
        std::string winName = "debug" + std::to_string(i);
        std::cout << "using winName " << winName << std::endl;
        cv::namedWindow(winName);
        cv::moveWindow(winName, 300, 100);
        simpleChromosome chromosome1(numCircles, minRadius, maxRadius, xRes, yRes);
        chromosome1.draw(canvas, winName);
    }

//    cv::imshow(winName, canvas);
    cv::waitKey();

    return 0;
}