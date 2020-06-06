#include <iostream>
#include "classes/shapes/circle.h"
#include "classes/individual/simpleIndividual.h"
#include "classes/population/simplePopulation.h"

#include "opencv2/core/mat.hpp"
#include "opencv2/highgui.hpp"

#include <chrono>
#include <mutex>
#include <cuda_profiler_api.h>

#include "cudaCircles.cuh"

using namespace vidga;
using namespace std::chrono_literals;

int main() {
    float **gpuBuf;
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
    auto numSubpixels = 3 * xRes * yRes;

//    auto targetCanvas = cv::Mat(yRes, xRes, CV_8UC3, cv::Scalar(255, 255, 255));
    const std::string targetWinName = "<= TARGET =>";
    cv::namedWindow(targetWinName);
    cv::imshow(targetWinName, img);

    cv::Mat imgForGpu;
    img.convertTo(imgForGpu, CV_32FC3, 1 / 255.f);
    float3 *imgGpu = cuda::getWhiteGpuMat(xRes, yRes);
    cudaMemcpy(imgGpu, imgForGpu.data, numSubpixels * sizeof(float), cudaMemcpyHostToDevice);

    // Create initial population
    auto population = std::make_shared<simplePopulation>(24, xRes, yRes, 150, false);

    const std::string firstItrWinName = "first iter";
    cv::namedWindow(firstItrWinName);
    float3 *canvas1 = cuda::getWhiteGpuMat(xRes, yRes);
    population->drawBest(canvas1);

    auto cpuCanvasData1 = new float[numSubpixels]();
    cudaDeviceSynchronize();
    cudaMemcpy(cpuCanvasData1, canvas1, numSubpixels * sizeof(float), cudaMemcpyDeviceToHost);
    auto canvas1Cpu = cv::Mat(yRes, xRes, CV_32FC3, cpuCanvasData1);
    cv::imshow(firstItrWinName, canvas1Cpu);

    auto generations = 2500;
    std::mutex mutex;

    auto bestPop = population;
    volatile bool threadsActive = true;
#ifndef __APPLE__
    auto drawThread = std::thread([&]() {
        const std::string current = "current";
        cv::namedWindow(current);
        const std::string best = "best";
        cv::namedWindow(best);
        const std::string diff = "diff";
        cv::namedWindow(diff);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-noreturn"
#pragma ide diagnostic ignored "EndlessLoop"
        auto canvasBestGpu = cuda::getWhiteGpuMat(xRes, yRes);
        auto canvasCurrGpu = cuda::getWhiteGpuMat(xRes, yRes);
        while (threadsActive) {
            auto canvas = cv::Mat(yRes, xRes, CV_8UC3, cv::Scalar(255, 255, 255));
            auto canvas2 = cv::Mat(yRes, xRes, CV_8UC3, cv::Scalar(255, 255, 255));

            cv::Mat canvasBestCpu, canvasCurrCpu, canvasBestDiff;
            {
                std::lock_guard<std::mutex> lock(mutex);

                cuda::setGpuMatTo(canvasBestGpu, xRes, yRes, 0x00);
                bestPop->drawBest(canvasBestGpu);
                auto cpuCanvasDataBest = new float[numSubpixels]();
                cudaDeviceSynchronize();
                cudaMemcpy(cpuCanvasDataBest, canvasBestGpu, numSubpixels * sizeof(float), cudaMemcpyDeviceToHost);
                canvasBestCpu = cv::Mat(yRes, xRes, CV_32FC3, cpuCanvasDataBest);

                cuda::setGpuMatTo(canvasCurrGpu, xRes, yRes, 0x00);
                population->drawBest(canvasCurrGpu);
                auto cpuCanvasDataCurr = new float[numSubpixels]();
                cudaDeviceSynchronize();
                cudaMemcpy(cpuCanvasDataCurr, canvasCurrGpu, numSubpixels * sizeof(float), cudaMemcpyDeviceToHost);
                canvasCurrCpu = cv::Mat(yRes, xRes, CV_32FC3, cpuCanvasDataCurr);

                cv::absdiff(canvasCurrCpu, imgForGpu, canvasBestDiff);
            }

            cv::imshow(best, canvasBestCpu);
            cv::imshow(current, canvasCurrCpu);
            cv::imshow(diff, canvasBestDiff);
            cv::waitKey(5000);
        }
#pragma clang diagnostic pop
    });

    drawThread.detach();
#endif

    int i = 0, prevI = 0;

    auto statusThread = std::thread([&i, &prevI, &threadsActive]() {
        while (threadsActive) {
            std::cout << "Speed: " << (i - prevI) / 5 << " Gen/s" << std::endl;
            prevI = i;
            std::this_thread::sleep_for(5s);
        }
    });

    statusThread.detach();

    for (i = 0; i < generations; i++) {
        cudaProfilerStart();
        population->sortByScore(imgGpu);

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
//    cv::waitKey();
    cudaProfilerStop();
    threadsActive = false;
    if (drawThread.joinable()) {
        drawThread.join();
    }
    if (statusThread.joinable()) {
        statusThread.join();
    }
    return 0;
}
