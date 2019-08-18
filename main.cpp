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
    // Load and display target image
    auto img = cv::imread("/Users/Bunk/Downloads/GA/mona.png");
    auto xRes = img.cols;
    auto yRes = img.rows;
    auto targetCanvas = cv::Mat(yRes, xRes, CV_8UC3, cv::Scalar(255, 255, 255));
    const std::string targetWinName = "<= TARGET =>";
    cv::namedWindow(targetWinName);
    cv::imshow(targetWinName, img);

    // Create initial population
    auto population = std::make_shared<simplePopulation>(40, xRes, yRes, 3.5);

    const std::string firstItrWinName = "first iter";
    cv::namedWindow(firstItrWinName);
    auto canvas1 = cv::Mat(yRes, xRes, CV_8UC3, cv::Scalar(255, 255, 255));
    population->getIndividuals()[0]->draw(canvas1);
    cv::imshow(firstItrWinName, canvas1);

    auto generations = 25000;
    for (auto i = 0; i < generations; i++) {
        population->sortByScore(targetCanvas);
        if (i % 100 == 0) {
            std::cout << "Generation [" << i+1 << " / " << generations << "] score is ["
                      << population->getIndividuals()[0]->getScore() << "]." << std::endl;
        }
        population = population->nextGeneration();
    }

    const std::string afterIterWinName = "after iters";
    cv::namedWindow(firstItrWinName);
    auto canvas2 = cv::Mat(yRes, xRes, CV_8UC3, cv::Scalar(255, 255, 255));
    population->getIndividuals()[0]->draw(canvas2);
    cv::imshow(afterIterWinName, canvas2);




    cv::waitKey();

    return 0;
}