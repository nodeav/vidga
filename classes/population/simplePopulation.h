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
        simplePopulation(uint32_t popSize, uint32_t xRes, uint32_t yRes, float circleAmountFactor = 2.5,
                         float minSizeFactor=0.02, float maxSizeFactor=0.3);


        const std::vector<std::shared_ptr<simpleIndividual>> getIndividuals() const;
        const void sortByScore(cv::Mat &target);
        std::shared_ptr<simplePopulation> nextGeneration();

    private:
        std::vector<std::shared_ptr<simpleIndividual>> individuals;
        uint32_t imgResX, imgResY;
        ucoor_t minSideLen, maxSideLen;

        // TODO: make this a single array of structs
        std::array<std::thread, 8> threadPool;
        std::array<std::unique_ptr<cv::Mat>, 8> canvasPool;
        std::array<std::unique_ptr<cv::Mat>, 8> scratchCanvasPool;
        void addIndividual(std::shared_ptr<simpleIndividual> individual);
    };
}


#endif //VIDGA_SIMPLEPOPULATION_H
