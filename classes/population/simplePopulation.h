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
                         float minSizeFactor = 0.001, float maxSizeFactor = 0.2);

        std::vector<std::shared_ptr<simpleIndividual>> getIndividuals() const;

        void sortByScore(float3 *target);

        std::shared_ptr<simplePopulation> nextGeneration();

    private:
        std::vector<std::shared_ptr<simpleIndividual>> individuals;
        uint32_t imgResX, imgResY;
        ucoor_t minSideLen, maxSideLen;
        uint16_t individualSize;
    public:
        uint16_t getIndividualSize() const;

        void drawBest(float3 *canvas) const;

    private:
        ThreadPool threadPool{24};
        std::vector<std::unique_ptr<cv::Mat>> canvasPool;
        std::vector<std::unique_ptr<cv::Mat>> scratchCanvasPool;

        void addIndividual(const std::shared_ptr<simpleIndividual> &individual);

        float **circlesMap;
    };
}


#endif //VIDGA_SIMPLEPOPULATION_H
