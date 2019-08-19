//
// Created by Nadav Eidelstein on 03/08/2019.
//
#include "simpleIndividual.h"

namespace vidga {
    std::vector<std::shared_ptr<shape>> &simpleIndividual::getShapesMut() {
        return shapes;
    }

    const std::vector<std::shared_ptr<shape>> &simpleIndividual::getShapes() const {
        return shapes;
    }

    simpleIndividual::simpleIndividual(size_t size, ucoor_t sideLengthMin, ucoor_t sideLengthMax,
                                       ucoor_t xMax, ucoor_t yMax) {
        shapes.reserve(size);
        for (auto i = 0; i < size; i++) {
            auto c = std::make_shared<circle>();
            c->setRandomEverything(sideLengthMin, sideLengthMax, xMax, yMax);
            shapes.push_back(c);
        }
    }

    void simpleIndividual::draw(cv::Mat &canvas) const {
        int i = 0;
        for (auto const &circle : shapes) {
            if (circle == nullptr) {
                std::cout << "circle #" << i++ << " is null!" << std::endl;
                continue;
            }
            const auto pt = cv::Point(circle->getCenter());
            cv::circle(canvas, pt, circle->getWidth(), cv::Scalar(circle->getColor()), -1);
        }
    }

    const auto getBit = [](int bits, int index) {
        return (bits >> index) & 1;
    };

    const auto genRandomInt() {
        return genRandom(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    }

    std::shared_ptr<simpleIndividual> simpleIndividual::randMerge(std::shared_ptr<simpleIndividual> src, ucoor_t sideLengthMin,
                                                   ucoor_t sideLengthMax, ucoor_t xMax, ucoor_t yMax) {
        auto dst = std::make_shared<simpleIndividual>(shapes.size(), sideLengthMin, sideLengthMax, xMax, yMax);
        auto& dstShapes = dst->getShapesMut();

        auto& srcShapes = src->getShapesMut();
        dstShapes.reserve(src->getShapes().size());

        // We only need 1 bit of randomness per decision
        const auto bitsPerInt = sizeof(int) * 8;
        const auto intsOfRandomness = static_cast<int>(dstShapes.size() / bitsPerInt + 1);

        for (auto i = 0; i < intsOfRandomness; i++) {
            auto oneInt = genRandomInt();
            auto idx = i * bitsPerInt;
            for (int j = 0; j < bitsPerInt && idx < dstShapes.size(); j++, idx++) {
                if (getBit(oneInt, j)) {
                    *dstShapes[idx] = *srcShapes[idx];
                } else {
                    *dstShapes[idx] = *shapes[idx];
                }

                dstShapes[idx]->mutate(0.001, xMax, yMax, sideLengthMin, sideLengthMax);
            }
        }
        return dst;
    }

    void simpleIndividual::calcAndSetScore(cv::Mat& target, cv::Mat& canvas) {
        draw(canvas);
        typedef cv::Point3_<uint8_t> Pixel;
        double newScore = 0;
//        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        for (int r = 0; r < target.rows; ++r) {
            Pixel* targetPtr = target.ptr<Pixel>(r, 0);
            Pixel* canvasPtr = canvas.ptr<Pixel>(r, 0);
            const Pixel* ptr_end = targetPtr + target.cols;
            while (targetPtr != ptr_end) {
                ++targetPtr;
                ++canvasPtr;
                newScore += abs(targetPtr->x - canvasPtr->x);
                newScore += abs(targetPtr->y - canvasPtr->y);
                newScore += abs(targetPtr->z - canvasPtr->z);
            }
        }
        score = static_cast<float>(newScore / (canvas.total() * canvas.channels()));

        if (score == 0) {
            std::cout << "score is 0 mofo" << std::endl;
        }
//        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    }

    float simpleIndividual::getScore() const {
        return score;
    }
}