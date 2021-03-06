#include "simplePopulation.h"
#include "cudaCircles.cuh"

namespace vidga {
    simplePopulation::simplePopulation(uint32_t popSize, uint32_t xRes, uint32_t yRes, uint16_t individualSize_,
                                       bool skipCircleMapsInit, float minSizeFactor, float maxSizeFactor) {

        imgResX = xRes;
        imgResY = yRes;

        auto avg = (xRes + yRes) / 2;
        minSideLen = std::max(1u, static_cast<ucoor_t>(minSizeFactor * avg));
        maxSideLen = static_cast<ucoor_t>(maxSizeFactor * avg);

        if (!skipCircleMapsInit) {
            std::cout << "initializing circle map with min " << minSideLen << " and max " << maxSideLen << "\n";
            vidga::cuda::initCircleMaps(minSideLen, maxSideLen, &draw_map);
            vidga::cuda::initCircleMaps1D(minSideLen, maxSideLen, &diff_map);
        }

        individualSize = individualSize_;

        individuals.reserve(popSize);
        for (auto i = 0; i < popSize; i++) {
            auto individual = std::make_shared<simpleIndividual>(individualSize, minSideLen, maxSideLen, xRes, yRes);
            individuals.push_back(individual);
        }
    }

    uint16_t simplePopulation::getIndividualSize() const {
        return individualSize;
    }

    std::vector<std::shared_ptr<simpleIndividual>> simplePopulation::getIndividuals() const {
        return individuals;
    }

    void simplePopulation::sortByScore(float3 *target) {
        static std::vector<std::future<void>> futures{individuals.size()};
        for (auto i = 0; i < individuals.size(); i++) {
            futures[i] = threadPool->enqueue([&](int i) {
                thread_local auto canvas = cuda::getWhiteGpuMat(imgResX, imgResY);
                cudaStream_t stream;
                auto tid = std::this_thread::get_id();
                auto cudaStream = cudaStreams.find(tid);
                if (cudaStream == cudaStreams.end()) {
                    cudaStreamCreate(&stream);
                    cudaStreams.insert({tid, stream});
                } else {
                    stream = cudaStream->second;
                }
                individuals[i]->calcAndSetScore(target, canvas, diff_map, stream);
                cuda::setGpuMatTo(canvas, imgResX, imgResY, 0.f);
            }, i);
        }

        for (auto &future : futures) {
            future.wait();
        }

        std::sort(individuals.begin(), individuals.end(), [](const auto &first, const auto &second) {
            return first->getScore() < second->getScore();
        });
    }

    std::shared_ptr<simplePopulation> simplePopulation::nextGeneration() {
        auto topIndividualsCutoff = static_cast<int>(individuals.size() * 0.2);
        auto result = std::make_shared<simplePopulation>(0, imgResX, imgResY, individualSize);
        result->threadPool = std::move(threadPool);
        result->draw_map = draw_map;
        result->diff_map = diff_map;
        result->cudaStreams = cudaStreams;
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

    void simplePopulation::addIndividual(const std::shared_ptr<simpleIndividual> &individual) {
        individuals.push_back(individual);
    }

    void simplePopulation::drawBest(float3 *canvas) const {
        individuals[0]->draw(canvas, draw_map);
    }

}
