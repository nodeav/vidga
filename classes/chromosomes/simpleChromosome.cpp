//
// Created by Nadav Eidelstein on 03/08/2019.
//
#include "simpleChromosome.h"

namespace vidga {
    std::vector<std::unique_ptr<shape>>& simpleChromosome::getShapesMut() {
        return shapes;
    }

    const std::vector<std::unique_ptr<shape>>& simpleChromosome::getShapes() const {
        return shapes;
    }

    simpleChromosome::simpleChromosome(size_t size, ucoor_t sideLengthMin, ucoor_t sideLengthMax,
                                       ucoor_t xMax, ucoor_t yMax) {
        shapes.reserve(size);
        for (auto i = 0; i < size; i++) {
            auto c = std::make_unique<circle>();
            c->setRandom(sideLengthMin, sideLengthMax, xMax, yMax);
            shapes.push_back(std::move(c));
        }
    }

    void simpleChromosome::draw(cv::Mat &canvas, std::string &windowName) const {
        const auto getColor = []() {
            return genRandom(0, 255);
        };

        const auto getColorScalar = [=]() {
            return cv::Scalar(getColor(), getColor(), getColor());
        };

        for (auto const &circle : shapes) {
            std::cout << "center, radius: " << circle->getCenter() << ", " << circle->getWidth() << std::endl;
            const auto pt = cv::Point(circle->getCenter());
            cv::circle(canvas, pt, circle->getWidth(), getColorScalar(), -1);
        }
    }

    const auto getBit = [](int bits, int index) {
        return (bits >> index) & 1;
    };

    const auto genRandomInt() {
        return genRandom(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    }

    void simpleChromosome::mutRandMerge(simpleChromosome &src) {
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
            for (auto j = 0; j < bitsPerInt && (i+j) < dstShapes.size(); j++) {
                if (getBit(oneInt, j)) {
                    dstShapes[i+j] = std::move(srcShapes[i+j]);
                }
            }
        }
//        srcShapes.clear();
    }
}
