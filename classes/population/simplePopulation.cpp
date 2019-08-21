//
// Created by Nadav Eidelstein on 07/08/2019.
//

#include "simplePopulation.h"

namespace vidga {

    simplePopulation::simplePopulation(uint32_t popSize, uint32_t xRes, uint32_t yRes, float circleAmountFactor,
                                              float minSizeFactor, float maxSizeFactor) {

        imgResX = xRes;
        imgResY = yRes;
        minSideLen = static_cast<ucoor_t>(minSizeFactor * xRes);
        maxSideLen = static_cast<ucoor_t>(maxSizeFactor * xRes);
        const float avgRadius = static_cast<ucoor_t>((minSideLen + maxSideLen) / 2);

        const auto avgCircleSize = (avgRadius * avgRadius * M_PI);
        auto numCircles = static_cast<size_t >(circleAmountFactor * xRes * yRes / avgCircleSize);

        individuals.reserve(popSize);
        for (auto i = 0; i < popSize; i++) {
            auto individual = std::make_shared<simpleIndividual>(numCircles, minSideLen, maxSideLen, xRes, yRes);
            individuals.push_back(individual);
        }
    }

    const std::vector<std::shared_ptr<simpleIndividual>> simplePopulation::getIndividuals() const {
        return individuals;
    }

    const void simplePopulation::sortByScore(cv::Mat &target) {

        // currently has to be a natural number
        auto numPerThread = individuals.size() / threadPool.size();

        for (auto i = 0; i < threadPool.size(); i++) {
            threadPool[i] = std::thread([&](int i) {
                cv::Mat canvas(target.rows, target.cols, target.type(), {255, 255, 255});
                cv::Mat scratchCanvas(target.rows, target.cols, CV_32F);
                const auto from = numPerThread * i;
                const auto to = numPerThread + from;
                for (auto j = from; j < to; j++) {
                    individuals[j]->calcAndSetScore(target, canvas, scratchCanvas);
                    canvas = cv::Scalar({255, 255, 255});
                }
            }, i);
        }

        for (auto& thread : threadPool) {
            thread.join();
        }

        std::sort(individuals.begin(), individuals.end(), [](const auto &first, const auto &second) {
            return first->getScore() < second->getScore();
        });
    }

    std::shared_ptr<simplePopulation> simplePopulation::nextGeneration() {
        auto topIndividualsCutoff = static_cast<int>(individuals.size() * 0.25);
        auto result = std::make_shared<simplePopulation>(0, imgResX, imgResY);

        auto getRandomIndex = [&]() {
            return genRandom(0, topIndividualsCutoff);
        };

        for (auto i = 0; i < individuals.size(); i++) {
            auto randIdx1 = getRandomIndex();
            auto randIdx2 = getRandomIndex();
            auto newIndividual = individuals[randIdx1]->randMerge(individuals[randIdx2],
                    minSideLen, maxSideLen, imgResX, imgResY);

            result->addIndividual(newIndividual);
        }

        return result;
    }

    void simplePopulation::addIndividual(std::shared_ptr<simpleIndividual> individual) {
        individuals.push_back(individual);
    }
}