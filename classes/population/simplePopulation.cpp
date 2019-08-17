//
// Created by Nadav Eidelstein on 07/08/2019.
//

#include "simplePopulation.h"

vidga::simplePopulation::simplePopulation(uint32_t popSize, uint32_t xRes, uint32_t yRes, float circleAmountFactor,
        float minSizeFactor, float maxSizeFactor) {

    const auto minRadius = static_cast<ucoor_t>(minSizeFactor * xRes);
    const auto maxRadius = static_cast<ucoor_t>(maxSizeFactor * xRes);
    const float avgRadius = static_cast<ucoor_t>((minRadius + maxRadius) / 2);

    const auto avgCircleSize = (avgRadius * avgRadius * M_PI);
    auto numCircles = static_cast<size_t >(circleAmountFactor * xRes * yRes / avgCircleSize);

    individuals.reserve(popSize);
    for (auto i = 0; i < popSize; i++) {
        auto individual = std::make_unique<simpleIndividual>(numCircles, minRadius, maxRadius, xRes, yRes);
        individuals.push_back(std::move(individual));
    }
}

const std::vector<std::unique_ptr<vidga::simpleIndividual>> &vidga::simplePopulation::getIndividuals() const {
    return individuals;
}

const void vidga::simplePopulation::sortByScore(cv::Mat &target) {
    cv::Mat canvas(target.rows, target.cols, target.type());
    for (auto& individual : individuals) {
        canvas = cv::Scalar({255, 255, 255});
        individual->calcAndSetScore(target, canvas);
    }
    std::sort(individuals.begin(), individuals.end(), [](const auto& first, const auto& second) {
        return first->getScore() < second->getScore();
    });
}
