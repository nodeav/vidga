//
// Created by Nadav Eidelstein on 03/08/2019.
//
#include "simpleIndividual.h"

namespace vidga {
    std::vector<std::unique_ptr<shape>>& simpleIndividual::getShapesMut() {
        return shapes;
    }

    const std::vector<std::unique_ptr<shape>>& simpleIndividual::getShapes() const {
        return shapes;
    }

    simpleIndividual::simpleIndividual(size_t size, ucoor_t sideLengthMin, ucoor_t sideLengthMax,
                                       ucoor_t xMax, ucoor_t yMax) {
        shapes.reserve(size);
        for (auto i = 0; i < size; i++) {
            auto c = std::make_unique<circle>();
            c->setRandom(sideLengthMin, sideLengthMax, xMax, yMax);
            shapes.push_back(std::move(c));
        }
    }

    void simpleIndividual::draw(cv::Mat &canvas, std::string &windowName) const {
        const auto getColor = []() {
            return genRandom(0, 255);
        };

        const auto getColorScalar = [=]() {
            return cv::Scalar(getColor(), getColor(), getColor());
        };

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

    void simpleIndividual::mutRandMerge(simpleIndividual &src) {
        auto& dstShapes = getShapesMut();
        auto& srcShapes = src.getShapesMut();
        dstShapes.reserve(src.getShapes().size());

        // We only need 1 bit of randomness per decision
        const auto bitsPerInt = sizeof(int) * 8;
        const auto intsOfRandomness = static_cast<int>(dstShapes.size() / bitsPerInt + 1);

        auto srcIt = srcShapes.begin();
        auto dstIt = dstShapes.begin();

        for (auto i = 0; i < intsOfRandomness; i++) {
            auto oneInt = genRandomInt();
            auto idx = i * bitsPerInt;
            for (int j = 0; j < bitsPerInt && idx < dstShapes.size(); j++, idx++) {
                if (getBit(oneInt, j)) {
                    dstShapes[idx] = std::move(srcShapes[idx]);
                }
            }
        }
//        srcShapes.clear();
    }
}
