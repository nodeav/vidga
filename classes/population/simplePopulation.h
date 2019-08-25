//
// Created by Nadav Eidelstein on 07/08/2019.
//

#ifndef VIDGA_SIMPLEPOPULATION_H
#define VIDGA_SIMPLEPOPULATION_H

#include <math.h>
#include <vector>
#include <thread>
#include "../individual/simpleIndividual.h"

namespace vidga {
    class simplePopulation {
    public:
        simplePopulation(uint32_t popSize, uint32_t xRes, uint32_t yRes, uint16_t individualSize,
                         float minSizeFactor=0.02, float maxSizeFactor=0.3);

        const std::vector<std::shared_ptr<simpleIndividual>> getIndividuals() const;
        const void sortByScore(cv::Mat &target);
        std::shared_ptr<simplePopulation> nextGeneration();

    private:
        std::vector<std::shared_ptr<simpleIndividual>> individuals;
        uint32_t imgResX, imgResY;
        ucoor_t minSideLen, maxSideLen;
        uint16_t individualSize;
    public:
        uint16_t getIndividualSize() const;

    private:
        // TODO: make this a single array of structs
        std::array<std::thread, 16> threadPool;
        std::array<std::unique_ptr<cv::Mat>, 16> canvasPool;
        std::array<std::unique_ptr<cv::Mat>, 16> scratchCanvasPool;
        void addIndividual(std::shared_ptr<simpleIndividual> individual);
    };
}


#endif //VIDGA_SIMPLEPOPULATION_H
