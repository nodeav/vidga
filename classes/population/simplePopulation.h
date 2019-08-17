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
        simplePopulation(uint32_t popSize, uint32_t xRes, uint32_t yRes, float circleAmountFactor,
                         float minSizeFactor=0.03, float maxSizeFactor=0.1);

        const std::vector<std::unique_ptr<simpleIndividual>>& getIndividuals() const;
        const void sortByScore(cv::Mat &target);

    private:
        std::vector<std::unique_ptr<simpleIndividual>> individuals;
    };
}


#endif //VIDGA_SIMPLEPOPULATION_H
