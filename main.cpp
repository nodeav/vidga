#include <iostream>
#include "classes/shapes/circle.h"
#include "classes/individual/simpleIndividual.h"
#include "classes/population/simplePopulation.h"

#include "opencv2/core/mat.hpp"
#include "opencv2/highgui.hpp"

#include <chrono>
#include <mutex>

using namespace vidga;
using namespace std::chrono_literals;

int main() {
    // Load and display target image
//    auto img = cv::imread("../target.png");

    auto videoPath = "/home/nadav/Pictures/giphy.gif";
    auto vid = cv::VideoCapture(videoPath);
    if (!vid.isOpened()) {
        std::cerr << "video '" << videoPath << "' not opened!" << std::endl;
        return EXIT_FAILURE;
    }

    auto width = vid.get(cv::CAP_PROP_FRAME_WIDTH);
    auto height = vid.get(cv::CAP_PROP_FRAME_HEIGHT);
    auto fps = vid.get(cv::CAP_PROP_FPS);

    std::cout << "Vid dimensions: " << width << "x" << height << std::endl;
    auto xRes = width;
    auto yRes = height;
    auto targetCanvas = cv::Mat(yRes, xRes, CV_8UC3, cv::Scalar(255, 255, 255));

    // Create initial population
    auto population = std::make_shared<simplePopulation>(48, xRes, yRes, 300);

    const std::string firstItrWinName = "first iter";
    cv::namedWindow(firstItrWinName);
    auto canvas1 = cv::Mat(yRes, xRes, CV_8UC3, cv::Scalar(255, 255, 255));
    population->getIndividuals()[0]->draw(canvas1);
    cv::imshow(firstItrWinName, canvas1);

    auto minGenerations = 1'000, maxGenerations = 5'000;
    auto goodEnoughScore = 10;
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

    cv::Mat frame;
    auto frame_idx = 0;
    const std::string targetWinName = "<= TARGET =>";
    cv::namedWindow(targetWinName);

    auto codec = cv::VideoWriter::fourcc('X', '2', '6', '4');
    cv::VideoWriter output("../output.mp4", codec, fps, cv::Size(width, height));

    if (!output.isOpened()) {
        std::cout << "Could not open output file!" << std::endl;
    }

    while (true) {
        vid >> frame;

        if (frame.empty()) {
            break;
        }

        frame_idx++;
        cv::imshow(targetWinName, frame);
        std::cout << "Frame #" << frame_idx << ":\n";
        for (i = 0; i < maxGenerations; i++) {
            population->sortByScore(frame);

            if (population->getIndividuals().front()->getScore() < bestPop->getIndividuals().front()->getScore()) {
                bestPop = population;
            }

            if (i % 100 == 0) {
                std::cout << "\tGeneration [" << i + 1 << " / " << maxGenerations << "] score is ["
                          << population->getIndividuals().front()->getScore() << "],"
                          << " and worst score is " << population->getIndividuals().back()->getScore()
                          << ". best population had score of " << bestPop->getIndividuals().front()->getScore()
                          << std::endl;
            }

            if (bestPop->getIndividuals().front()->getScore() < goodEnoughScore && i > minGenerations) {
                break;
            }

            std::lock_guard<std::mutex> lock(mutex);
            population = population->nextGeneration();
        }
        std::lock_guard<std::mutex> lock(mutex);
        bestPop->getIndividuals()[0]->draw(frame);
        output.write(frame);
    }

#ifdef __APPLE__
    const std::string best = "best match";
    cv::namedWindow(best);
    auto canvas2 = cv::Mat(yRes, xRes, CV_8UC3, cv::Scalar(255, 255, 255));
    population->getIndividuals()[0]->draw(canvas2);
    cv::imshow(best, canvas2);
#endif
    auto canvas = cv::Mat(yRes, xRes, CV_8UC3, cv::Scalar(255, 255, 255));
    bestPop->getIndividuals()[0]->draw(canvas);
    cv::imwrite("./best.png", canvas);
    cv::waitKey();

    return 0;
}
