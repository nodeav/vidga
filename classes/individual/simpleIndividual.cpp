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
                std::shared_ptr<shape>ptr;
                if (getBit(oneInt, j)) {
                    ptr = srcShapes[idx];
                } else {
                    ptr = shapes[idx];
                }
                dstShapes[idx]->setColor(ptr->getColor());
                dstShapes[idx]->setWidth(ptr->getWidth());
                dstShapes[idx]->setCenter(ptr->getCenter());
                dstShapes[idx]->mutate(0.01, xMax, yMax, sideLengthMin, sideLengthMax);
            }
        }
        return dst;
    }

    void simpleIndividual::calcAndSetScore(cv::Mat& target, cv::Mat& canvas, cv::Mat& dst) {
        draw(canvas);
        cv::absdiff(target, canvas, dst);
        cv::Scalar newScore = cv::sum(dst);
        score = (newScore.val[0] + newScore.val[1] + newScore.val[2]) / (canvas.total() * canvas.channels());
    }

    float simpleIndividual::getScore() const {
        return score;
    }
}
