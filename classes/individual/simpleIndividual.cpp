//
// Created by Nadav Eidelstein on 03/08/2019.
//
#include <mutex>

#include "util.h"
#include "simpleIndividual.h"
#include "cudaCircles.cuh"

namespace vidga {
    std::vector<circle> &simpleIndividual::getShapes() {
        return circles;
    }

    simpleIndividual::simpleIndividual(size_t size, ucoor_t sideLengthMin, ucoor_t sideLengthMax,
                                       ucoor_t xMax, ucoor_t yMax) {
        width = xMax;
        height = yMax;
        minMapRadius = sideLengthMin;
        circles.reserve(size);
        for (auto i = 0; i < size; i++) {
            circle c;
            c.setRandomEverything(sideLengthMin, sideLengthMax, xMax, yMax);
            circles.push_back(c);
        }
    }

    void simpleIndividual::draw(float3 *canvas, float **map) const {
//        cuda::drawManyUsingMapHostFn(canvas, width, height, map, minMapRadius, circles.data(), circles.size());
//        cudaDeviceSynchronize();
        for (auto const &circle : circles) {
            auto idx = circle.radius - minMapRadius;
            cuda::drawUsingMapHostFn(canvas, width, height, map[idx], circle);
        }
    }

    const auto getBit = [](int bits, int index) {
        return (bits >> index) & 1;
    };

    auto genRandomInt() {
        return genRandom(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    }

    std::shared_ptr<simpleIndividual>
    simpleIndividual::randMerge(const std::shared_ptr<simpleIndividual>& src, ucoor_t sideLengthMin,
                                ucoor_t sideLengthMax, ucoor_t xMax, ucoor_t yMax) {
        auto dst = std::make_shared<simpleIndividual>(circles.size(), sideLengthMin, sideLengthMax, xMax, yMax);

        dst->targetCopied = true;
        dst->targetCPU = targetCPU;
        dst->canvasCPU = canvasCPU;
        dst->targetMat = targetMat;
        dst->canvasMat = canvasMat;

        auto &dstShapes = dst->getShapes();

        auto &srcShapes = src->getShapes();
        dstShapes.reserve(src->getShapes().size());

        // We only need 1 bit of randomness per decision
        const auto bitsPerInt = sizeof(int) * 8;
        const auto intsOfRandomness = static_cast<int>(dstShapes.size() / bitsPerInt + 1);

        for (auto i = 0; i < intsOfRandomness; i++) {
            auto oneInt = genRandomInt();
            auto idx = i * bitsPerInt;
            for (int j = 0; j < bitsPerInt && idx < dstShapes.size(); j++, idx++) {
                circle cir;
                if (getBit(oneInt, j)) {
                    cir = srcShapes[idx];
                } else {
                    cir = circles[idx];
                }
                dstShapes[idx] = cir;
                dstShapes[idx].mutate(0.1, xMax, yMax, sideLengthMin, sideLengthMax);

                if (genRandom(0, 500) < 1) {
                    auto idx1 = genRandom(0, static_cast<int>(dstShapes.size() - 1));
                    auto idx2 = genRandom(0, static_cast<int>(dstShapes.size() - 1));
                    std::iter_swap(dstShapes.begin() + idx1, dstShapes.begin() + idx2);
                }
            }
        }
        return dst;
    }

#define gpu_check(e)    \
if (e != cudaSuccess) { \
    printf("cuda error - %d on %s:%d\n", e, __FILE__, __LINE__); \
    }

    void simpleIndividual::calcAndSetScore(float3 *target, float3 *canvas, float *circlesMap, cudaStream_t cudaStream) {
//        cv::Mat scratchPad(height, width, CV_32FC3);
        auto numSubpixels = width * height * 3;

        if (!targetCopied) {
//            cv::namedWindow("temp");
            targetCPU = new float[numSubpixels]();
            canvasCPU = new float[numSubpixels]();
            cudaMemcpyAsync(targetCPU, target, numSubpixels * sizeof(float), cudaMemcpyDeviceToHost, cudaStream);
            targetMat = cv::Mat(height, width, CV_32FC3, targetCPU);
            targetCopied = true;
        }

        cuda::calcDiffUsingMapHostFn(canvas, target, width, height, circlesMap, circles, minMapRadius, cudaStream);
        gpu_check(cudaStreamSynchronize(cudaStream));
//        draw(canvas, circlesMap);

        cudaMemcpyAsync(canvasCPU, canvas, numSubpixels * sizeof(float), cudaMemcpyDeviceToHost, cudaStream);
        canvasMat = cv::Mat(height, width, CV_32FC3, canvasCPU);
        gpu_check(cudaStreamSynchronize(cudaStream));

//        cv::imshow("temp", canvasMat);
//        cv::waitKey(0);

//        cv::absdiff(targetMat, canvasMat, scratchPad);
        cv::Scalar newScore = cv::sum(canvasMat);
        score = static_cast<float>((newScore.val[0] + newScore.val[1] + newScore.val[2]) /
                                   (targetMat.total() * targetMat.channels()));
    }

    float simpleIndividual::getScore() const {
        return score;
    }
}
