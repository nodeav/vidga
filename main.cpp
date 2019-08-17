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

    const auto target = simplePopulation(1, xRes, yRes, 2.5);
    auto targetCanvas = cv::Mat(yRes, xRes, CV_8UC3, cv::Scalar(255, 255, 255));
    const std::string targetWinName = "<= TARGET =>";
    target.getIndividuals()[0]->draw(targetCanvas);
    cv::namedWindow(targetWinName);
    cv::imshow(targetWinName, targetCanvas);

    auto population = simplePopulation(100, xRes, yRes, 2.5);

    auto i = 0;
    auto scratchCanvas = cv::Mat(yRes, xRes, CV_8UC3, cv::Scalar(255, 255, 255));
    auto scratchCanvas2 = cv::Mat(yRes, xRes, CV_8UC3, cv::Scalar(255, 255, 255));
    population.sortByScore(targetCanvas);
    std::string bestWindow = "bestIndividual";
    std::string worstWindow = "worstIndividual";
    cv::namedWindow(bestWindow);
    cv::namedWindow(worstWindow);
    auto&& best = population.getIndividuals().front();
    std::cout << "best score is " << best->getScore() << std::endl;

    auto&& worst = population.getIndividuals().back();
    std::cout << "worst score is " << worst->getScore() << std::endl;

    auto font = cv::FONT_HERSHEY_SIMPLEX;

    std::cout << "going to draw 'best'..." << std::endl;
    best->draw(scratchCanvas);
    cv::putText(scratchCanvas, "Score: " + std::to_string(best->getScore()), {100, 100}, font, 1, {0, 0, 0}, 3, cv::LINE_AA);
    cv::imshow(bestWindow, scratchCanvas);
    std::cout << "drew 'best'!" << std::endl;

    std::cout << "going to draw 'worst'..." << std::endl;
    worst->draw(scratchCanvas2);
    cv::putText(scratchCanvas2, "Score: " + std::to_string(worst->getScore()), {100, 100}, font, 1, {0, 0, 0}, 3, cv::LINE_AA);
    cv::imshow(worstWindow, scratchCanvas2);
    std::cout << "drew 'worst'!" << std::endl;

    cv::waitKey();

    return 0;
}