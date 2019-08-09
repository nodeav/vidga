//
// Created by Nadav Eidelstein on 30/07/2019.
//

#include "circle.h"
#include <iostream>

namespace vidga {
    ucoor_t circle::getWidth() const {
        return radius;
    }

    ucoor_t circle::getHeight() const {
        return radius;
    }

    bgr_color_t circle::getColor() const {
        return color;
    }

    void circle::setHeight(ucoor_t newHeight) {
        radius = newHeight;
    }

    void circle::setWidth(ucoor_t newWidth) {
        radius = newWidth;
    }

    void circle::setColor(vidga::bgr_color_t newColor) {
        color = newColor;
    }

    bool circle::contains(coors c) const {
        return center.distanceFrom(c) <= radius;
    }

    circle::circle(coors center_, ucoor_t radius_)
        : shape(center_)
        , radius(radius_) {
    }

    circle::circle(ucoor_t radius_) : circle({0, 0}, radius_) {
        std::cout << "radius is " << radius << " and coors are {" << center.x << ", " << center.y << "}" << std::endl;

    }

    void circle::setRandom(ucoor_t sideLengthMin, ucoor_t sideLengthMax, ucoor_t xMax, ucoor_t yMax) {
        center = coors::generateRandom(xMax, yMax);
        radius = static_cast<ucoor_t>(genRandom(sideLengthMin, sideLengthMax));
        color = {
                static_cast<uint8_t>(genRandom(0, 255)),
                static_cast<uint8_t>(genRandom(0, 255)),
                static_cast<uint8_t>(genRandom(0, 255))
        };
    }

    circle::circle() : circle({0, 0}, 0) {

    }

    coors circle::getCenter() const {
        return center;
    }

    void circle::mutate(float chance, ucoor_t xMax, ucoor_t yMax, ucoor_t sizeMax) {

        const auto getNumberWithinRange = [](auto var, auto from, auto to) {
            return std::max(from, std::min(var, to));
        };
        auto finalChance = static_cast<int>(100 / getNumberWithinRange(chance, 0.f, 100.f));


        const auto mutateVar = [=](auto& var, auto maxValue) {
            auto shouldMutate = genRandom(0, finalChance) == 1;
            if (shouldMutate) {
                const auto max = maxValue / 2;
                const auto min = -1 * max;
                var += genRandom(min, max);
                var = getNumberWithinRange(static_cast<int>(var), 0, static_cast<int>(maxValue));
                var = var % maxValue;
                var = var < 0 ? 0 : var;
            }
        };

        mutateVar(radius, sizeMax);
        mutateVar(center.x, xMax);
        mutateVar(center.y, yMax);
        mutateVar(color.r, 255);
        mutateVar(color.g, 255);
        mutateVar(color.b, 255);
    }
}