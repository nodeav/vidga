//
// Created by Nadav Eidelstein on 03/08/2019.
//
#include <mutex>

#include "util.h"
#include "simpleIndividual.h"
#include "cudaCircles.cuh"

namespace vidga {
    std::vector<circle> &simpleIndividual::getShapesMut() {
        return circles;
    }

    simpleIndividual::simpleIndividual(size_t size, ucoor_t sideLengthMin, ucoor_t sideLengthMax,
                                       ucoor_t xMax, ucoor_t yMax) {
        width = xMax;
        height = yMax;
        circles.reserve(size);
        for (auto i = 0; i < size; i++) {
            circle c;
            c.setRandomEverything(sideLengthMin, sideLengthMax, xMax, yMax);
            circles.push_back(c);
        }
    }

    void simpleIndividual::draw(float3 *canvas, float **map) const {
        int i = 0;
        for (auto const &circle : circles) {
            cuda::drawUsingMapHostFn(canvas, width, height, map, circle);
        }
    }

    const auto getBit = [](int bits, int index) {
        return (bits >> index) & 1;
    };

    const auto genRandomInt() {
        return genRandom(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    }

    std::shared_ptr<simpleIndividual>
    simpleIndividual::randMerge(std::shared_ptr<simpleIndividual> src, ucoor_t sideLengthMin,
                                ucoor_t sideLengthMax, ucoor_t xMax, ucoor_t yMax) {
        auto dst = std::make_shared<simpleIndividual>(circles.size(), sideLengthMin, sideLengthMax, xMax, yMax);
        auto &dstShapes = dst->getShapesMut();

        auto &srcShapes = src->getShapesMut();
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
                dstShapes[idx].mutate(0.5, xMax, yMax, sideLengthMin, sideLengthMax);

                if (genRandom(0, 50) < 1) {
                    auto idx1 = genRandom(0, static_cast<int>(dstShapes.size() - 1));
                    auto idx2 = genRandom(0, static_cast<int>(dstShapes.size() - 1));
                    std::iter_swap(dstShapes.begin() + idx1, dstShapes.begin() + idx2);
                }
            }
        }
        return dst;
    }

    void simpleIndividual::calcAndSetScore(float3 *target, float3 *canvas, float **circlesMap) {
        draw(canvas, circlesMap);
//        cv::absdiff(target, canvas, circlesMap);
//        cv::Scalar newScore = cv::sum(circlesMap);
//        score = static_cast<float>((newScore.val[0] + newScore.val[1] + newScore.val[2]) /
//                                   (canvas.total() * canvas.channels()));
    }

    float simpleIndividual::getScore() const {
        return score;
    }
}
