#include "simplePopulation.h"
#include "cudaCircles.cuh"

namespace vidga {
    simplePopulation::simplePopulation(uint32_t popSize, uint32_t xRes, uint32_t yRes, uint16_t individualSize_,
                                       float minSizeFactor, float maxSizeFactor) {

        imgResX = xRes;
        imgResY = yRes;

        auto avg = (xRes + yRes) / 2;
        minSideLen = static_cast<ucoor_t>(minSizeFactor * avg);
        maxSideLen = static_cast<ucoor_t>(maxSizeFactor * avg);

        vidga::cuda::initCircleMaps(minSideLen, maxSideLen, circlesMap);

        individualSize = individualSize_;

/*        std::cout << "Using: minSideLen=" << minSideLen <<
		     ", maxSideLen =" << maxSideLen <<
		     ", with " << individualSize << " individuals" << std::endl;
*/
        individuals.reserve(popSize);
        for (auto i = 0; i < popSize; i++) {
            auto individual = std::make_shared<simpleIndividual>(individualSize, minSideLen, maxSideLen, xRes, yRes);
            individuals.push_back(individual);
        }

        for (auto i = 0; i < canvasPool.size(); i++) {
            canvasPool[i] = std::make_unique<cv::Mat>(yRes, xRes, CV_8UC3, cv::Scalar{255, 255, 255});
            scratchCanvasPool[i] = std::make_unique<cv::Mat>(yRes, xRes, CV_32F);
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
            futures[i] = threadPool.enqueue([&](int i) {
                static thread_local cv::Mat canvas(imgResY, imgResX, CV_8UC3, cv::Scalar{255, 255, 255});
                static thread_local cv::Mat scratchCanvas(imgResY, imgResX, CV_32F);;
                individuals[i]->calcAndSetScore(target, canvas, circlesMap);
                canvas.setTo(cv::Scalar({255, 255, 255}));
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
