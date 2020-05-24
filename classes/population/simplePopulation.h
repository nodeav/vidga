//
// Created by Nadav Eidelstein on 07/08/2019.
//

#ifndef VIDGA_SIMPLEPOPULATION_H
#define VIDGA_SIMPLEPOPULATION_H

#include <cmath>
#include <vector>
#include <thread>
#include "../individual/simpleIndividual.h"
#include "util.h"

namespace vidga {
    class simplePopulation {
    public:
        simplePopulation(uint32_t popSize, uint32_t xRes, uint32_t yRes, uint16_t individualSize,
                         float minSizeFactor=0, float maxSizeFactor=0.5);

        std::vector<std::shared_ptr<simpleIndividual>> getIndividuals() const;
        void sortByScore(cv::Mat &target);
        std::shared_ptr<simplePopulation> nextGeneration();

    private:
        std::vector<std::shared_ptr<simpleIndividual>> individuals;
        uint32_t imgResX, imgResY;
        ucoor_t minSideLen, maxSideLen;
        uint16_t individualSize;
    public:
        uint16_t getIndividualSize() const;

    private:
        std::unique_ptr<ThreadPool> threadPool = std::make_unique<ThreadPool>(24);
        std::vector<std::unique_ptr<cv::Mat>> canvasPool;
        std::vector<std::unique_ptr<cv::Mat>> scratchCanvasPool;
        void addIndividual(std::shared_ptr<simpleIndividual> individual);
    };
}


#endif //VIDGA_SIMPLEPOPULATION_H
