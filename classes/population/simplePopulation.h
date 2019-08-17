//
// Created by Nadav Eidelstein on 07/08/2019.
//

#ifndef VIDGA_SIMPLEPOPULATION_H
#define VIDGA_SIMPLEPOPULATION_H

#include <math.h>
#include <vector>
#include "../individual/simpleIndividual.h"

namespace vidga {
    class simplePopulation {
    public:
        simplePopulation(uint32_t popSize, uint32_t xRes, uint32_t yRes, float circleAmountFactor = 2.5,
                         float minSizeFactor=0.03, float maxSizeFactor=0.1);


        const std::vector<std::shared_ptr<simpleIndividual>> getIndividuals() const;
        const void sortByScore(cv::Mat &target);
        std::shared_ptr<simplePopulation> nextGeneration();

    private:
        std::vector<std::shared_ptr<simpleIndividual>> individuals;
        uint32_t imgResX, imgResY;
        ucoor_t minSideLen, maxSideLen;

        void addIndividual(std::shared_ptr<simpleIndividual> shared_ptr);
    };
}


#endif //VIDGA_SIMPLEPOPULATION_H
