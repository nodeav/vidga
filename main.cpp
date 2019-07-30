#include <iostream>
#include <vector>
#include "classes/shapes/circle.h"

#include "opencv2/core/mat.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"


using namespace vidga;

void checkIfContains(circle &circle1, coors c) {
    std::string containsStr = circle1.contains(c) ? "contains" : "does not contain";
    std::cout << "C " << containsStr << " " << c << std::endl;
}

int main() {
    circle circle1({250, 250}, 20);

    auto contains = [&circle1](coors c) {
        checkIfContains(circle1, c);
    };

    contains({100, 0});
    contains({20, 0});
    contains({0, 20});
    contains({10, 10});
    contains({13, 13});
    contains({14, 14});
    contains({15, 15});
//    circle1.setCenter({250, 250});
    contains(circle1.getCenter());

    auto canvas = cv::Mat(500, 500, CV_8UC3);

    auto pt = cv::Point(circle1.getCenter());
    cv::circle(canvas, pt, circle1.getHeight(), cv::Scalar(255, 50, 50), -1);

    auto winName = "debug";
    cv::namedWindow(winName);
    cv::moveWindow(winName, 300, 100);
    cv::imshow(winName, canvas);
    cv::waitKey();

    std::cout << "pt.x: " << pt.x << ", pt.y: " << pt.y << std::endl;
    return 0;
}