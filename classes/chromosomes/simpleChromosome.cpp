//
// Created by Nadav Eidelstein on 03/08/2019.
//
#include <memory>
#include "simpleChromosome.h"

namespace vidga {
//    std::vector<shape> simpleChromosome::getShapesMut() {
//        return shapes;
//    }
//
//    std::vector<shape> simpleChromosome::getShapes() const {
//        return shapes;
//    }

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

        for (auto const& circle : shapes) {
            const auto pt = cv::Point(circle->getCenter());
            cv::circle(canvas, pt, circle->getWidth(), getColorScalar(), -1);
        }
        cv::imshow(windowName, canvas);
    }
}
