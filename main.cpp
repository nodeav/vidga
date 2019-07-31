#include <iostream>
#include <vector>
#include "classes/shapes/circle.h"

#include "opencv2/core/mat.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <chrono>

using namespace vidga;

int main() {
    circle circle1({250, 250}, 70);
//    int i = 0;
//    auto contains = [&circle1, &i](coors c) {
//        std::string containsStr = circle1.contains(c) ? "contains" : "does not contain";
//        std::cout << "C " << containsStr << " " << c << std::endl;
//        if (circle1.contains(c)) {
//            i++;
//        }
//    };

    ucoor_t xRes = 640, yRes = 360;
    const auto minRadius = static_cast<int>((3.0/100) * xRes);
    const auto maxRadius = static_cast<int>((10.0/100) * xRes);
    const float avgRadius = (minRadius + maxRadius) / 2;

    const auto getRadius = [=]() {
        return genRandom(minRadius, maxRadius);
    };

    const auto getColor = []() {
        return genRandom(0, 255);
    };

    const auto getColorScalar = [=]() {
        return cv::Scalar(getColor(), getColor(), getColor());
    };

    const auto getCoor = [=]() {
        return coors::generateRandom(xRes, yRes);
    };

    auto canvas = cv::Mat(yRes, xRes, CV_8UC3);

    const auto avgCircleSize = (avgRadius * avgRadius * 3.14159265);
    const auto circleAmountFactor = 2.5;
    int numCircles = static_cast<int>(circleAmountFactor * xRes * yRes / avgCircleSize);
    std::cout << "Using " << numCircles << " Circles." << std::endl;
    std::cout << "Using " << avgCircleSize << " as avgCircleSize." << std::endl;

    auto renderTook = 0;

    for (auto i = 0; i < numCircles; i++) {
        const auto pt = cv::Point(getCoor());
        const auto radius = getRadius();
        const auto colorScalar = getColorScalar();

        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        cv::circle(canvas, pt, radius, colorScalar, -1);
        std::chrono::steady_clock::time_point finish = std::chrono::steady_clock::now();
        renderTook += std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
    }
    const auto tookus = renderTook / 1000;
    float tookSecs = tookus/1e6f;
    std::cout << "Drawing circles took " << tookus << "µs, which is " << tookus/numCircles << "µs per circle, or around "
        << 1/(tookSecs/numCircles) << " circles/s" << std::endl;

    auto winName = "debug";
    cv::namedWindow(winName);
    cv::moveWindow(winName, 300, 100);
    cv::imshow(winName, canvas);
    cv::waitKey();

    return 0;
}