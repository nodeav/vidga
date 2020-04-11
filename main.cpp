#include <iostream>
#include "classes/shapes/circle.h"
#include "classes/individual/simpleIndividual.h"
#include "classes/population/simplePopulation.h"

#include "opencv2/core/mat.hpp"
#include "opencv2/highgui.hpp"

#include <chrono>
#include <mutex>
#include <cuda/cudaCircles.cuh>

using namespace vidga;
using namespace std::chrono_literals;

int main() {
    float** gpuBuf;
    initCircleMaps(3, 7, gpuBuf);
    cv::waitKey(0);
    // Load and display target image
//     auto img = cv::imread("/home/nadav/Downloads/photo6003684971056836606.jpg");
    auto img = cv::imread("/home/nadav/Documents/GeneticAlgorithm/mona.png");
//    auto img = cv::imread("/home/nadav/Pictures/pc-principle.jpg");
//    auto img = cv::imread("/home/nadav/Pictures/vlcsnap-2020-03-27-00h45m02s240.png"); // 4K!!
//    auto img = cv::imread("/home/nadav/Pictures/ratatouille.640x268.2.png");

    std::cout << "rows: " << img.rows << " and cols " << img.cols << std::endl;
    auto xRes = img.cols;
    auto yRes = img.rows;
    auto targetCanvas = cv::Mat(yRes, xRes, CV_8UC3, cv::Scalar(255, 255, 255));
    const std::string targetWinName = "<= TARGET =>";
    cv::namedWindow(targetWinName);
    cv::imshow(targetWinName, img);

    // Create initial population
    auto population = std::make_shared<simplePopulation>(60, xRes, yRes, 250, 0.001, 0.1);

    const std::string firstItrWinName = "first iter";
    cv::namedWindow(firstItrWinName);
    auto canvas1 = cv::Mat(yRes, xRes, CV_8UC3, cv::Scalar(255, 255, 255));
    population->getIndividuals()[0]->draw(canvas1);
    cv::imshow(firstItrWinName, canvas1);

    auto generations = 250000000;
    std::mutex mutex;

    auto bestPop = population;
#ifndef __APPLE__
    auto drawThread = std::thread([&population, &bestPop, &xRes, &yRes, &mutex]() {
        const std::string current = "current";
        cv::namedWindow(current);
        const std::string best = "best";
        cv::namedWindow(best);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-noreturn"
        while (true) {
            auto canvas = cv::Mat(yRes, xRes, CV_8UC3, cv::Scalar(255, 255, 255));
            auto canvas2 = cv::Mat(yRes, xRes, CV_8UC3, cv::Scalar(255, 255, 255));

            {
                std::lock_guard<std::mutex> lock(mutex);
                bestPop->getIndividuals()[0]->draw(canvas);
                population->getIndividuals()[0]->draw(canvas2);
            }

            cv::imshow(best, canvas);
            cv::imshow(current, canvas2);
            cv::waitKey(1000);
        }
#pragma clang diagnostic pop
    });

    drawThread.detach();
#endif

    int i = 0, prevI = 0;

    auto statusThread = std::thread([&i, &prevI]() {
        while (true) {
            std::cout << "Speed: " << (i - prevI) / 5 << " Gen/s" << std::endl;
            prevI = i;
            std::this_thread::sleep_for(5s);
        }
    });

    statusThread.detach();

    for (i = 0; i < generations; i++) {
        population->sortByScore(img);

        if (population->getIndividuals().front()->getScore() < bestPop->getIndividuals().front()->getScore()) {
            bestPop = population;
        }

        if (i % 100 == 0) {
            std::cout << "Generation [" << i + 1 << " / " << generations << "] score is ["
                      << population->getIndividuals().front()->getScore() << "],"
                      << " and worst score is " << population->getIndividuals().back()->getScore()
                      << ". best population had score of " << bestPop->getIndividuals().front()->getScore()
                      << std::endl;
        }

        std::lock_guard<std::mutex> lock(mutex);
        population = population->nextGeneration();
    }

#ifdef __APPLE__
    const std::string best = "best match";
    cv::namedWindow(best);
    auto canvas2 = cv::Mat(yRes, xRes, CV_8UC3, cv::Scalar(255, 255, 255));
    population->getIndividuals()[0]->draw(canvas2);
    cv::imshow(best, canvas2);
#endif
    cv::waitKey();

    return 0;
}
