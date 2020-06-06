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
            : shape(center_), radius(radius_) {
    }

    circle::circle(ucoor_t radius_) : circle({0, 0}, radius_) {
        std::cout << "radius is " << radius << " and coors are {" << center.x << ", " << center.y << "}" << std::endl;

    }

    void circle::setRandomEverything(ucoor_t sideLengthMin, ucoor_t sideLengthMax, ucoor_t xMax, ucoor_t yMax) {
        center = coors::generateRandom(xMax, yMax);
        radius = static_cast<ucoor_t>(genRandom(sideLengthMin, sideLengthMax));
        color = {
                genRandom(0.f, 1.f),
                genRandom(0.f, 1.f),
                genRandom(0.f, 1.f),
                genRandom(0.f, 0.7f)
        };
    }

    circle::circle() : circle({0, 0}, 0) {

    }

    coors circle::getCenter() const {
        return center;
    }

    void circle::mutate(float chance, ucoor_t xMax, ucoor_t yMax, ucoor_t sizeMin, ucoor_t sizeMax) {

        if (chance == 0.f) {
            return;
        }

        const auto getNumberWithinRange = [](auto var, auto from, auto to) {
            return std::max(from, std::min(var, to));
        };

        auto shouldMutate = [&]() { return genRandom(0, 100.f) < chance; };

        if (shouldMutate()) {
            radius = static_cast<ucoor_t>(genRandom(sizeMin, sizeMax));
        }

        if (shouldMutate()) {
            center.x = static_cast<ucoor_t>(genRandom(0, xMax));
        }

        if (shouldMutate()) {
            center.y = static_cast<ucoor_t>(genRandom(0, yMax));
        }

        if (shouldMutate()) {
            color.x = genRandom(0, 1);
        }

        if (shouldMutate()) {
            color.y = genRandom(0, 1);
        }

        if (shouldMutate()) {
            color.y = genRandom(0, 1);
        }

        if (shouldMutate()) {
            color.w = genRandom(0, 0.7);
        }
    }
//
//    circle &circle::operator=(const circle &rhs) noexcept {
//        center = rhs.center;
//        radius = rhs.radius;
//        color = rhs.color;
//        return *this;
//    }
}