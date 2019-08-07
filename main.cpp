#include <iostream>
#include <vector>
#include "classes/shapes/circle.h"
#include "classes/individual/simpleIndividual.h"
#include "classes/population/simplePopulation.h"

#include "opencv2/core/mat.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <chrono>

using namespace vidga;

int main() {
    ucoor_t xRes = 480, yRes = 270;
    /*const auto minRadius = static_cast<ucoor_t>((3.0/100) * xRes);
    const auto maxRadius = static_cast<ucoor_t>((10.0/100) * xRes);
    const float avgRadius = static_cast<ucoor_t>((minRadius + maxRadius) / 2);

    const auto avgCircleSize = (avgRadius * avgRadius * 3.14159265);
    const auto circleAmountFactor = 2.5;
    auto numCircles = static_cast<size_t >(circleAmountFactor * xRes * yRes / avgCircleSize);

    std::vector<simpleIndividual> individuals;
    individuals.reserve(3);

    for (auto i = 0; i < 3; i++) {
        auto canvas = cv::Mat(yRes, xRes, CV_8UC3, cv::Scalar(0, 0, 0));
        std::string winName = "individual #" + std::to_string(i);
        std::cout << "using winName " << winName << std::endl;

        if (i < 2) {
            individuals.emplace_back(numCircles, minRadius, maxRadius, xRes, yRes);
        } else {
            winName = "merged";
            individuals[0].mutRandMerge(individuals[1]);
        }

        cv::namedWindow(winName);
        cv::moveWindow(winName, xRes*(i%2), (yRes+50)*(i<2?1:2));

        std::cout << "Going to draw a " << individuals[i%2].getShapes().size() << "-long vector of shapes" << std::endl;
        // Weird things happen without this line
        individuals[i%2].draw(canvas, winName);
        cv::imshow(winName, canvas);
        canvas.release();
    }
     */
    const auto population = simplePopulation(3, xRes, yRes, 2.5);
    auto i = 0;
    for (auto&& individual : population.getIndividuals()) {
        auto canvas = cv::Mat(yRes, xRes, CV_8UC3, cv::Scalar(255, 255, 255));
        std::string winName = "individual #" + std::to_string(i++);
        cv::namedWindow(winName);
        individual->draw(canvas, winName);
        cv::imshow(winName, canvas);
    }
    cv::waitKey();

    return 0;
}