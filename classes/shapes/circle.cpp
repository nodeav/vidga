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

    const ucoor_t circle::getRadius() const {
        return radius;
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
                static_cast<uint8_t>(genRandom(0, 255)),
                static_cast<uint8_t>(genRandom(0, 255)),
                static_cast<uint8_t>(genRandom(0, 255))
        };
    }

    circle::circle() : circle({0, 0}, 0) {

    }

    circle::circle(const circle& c) : shape(c.center) {
        this->center = c.center;
        this->radius = c.radius;
        this->color = c.color;
    }

    coors circle::getCenter() const {
        return center;
    }

    void circle::mutate(float chance, ucoor_t xMax, ucoor_t yMax, ucoor_t sizeMin, ucoor_t sizeMax) {

        if (!chance) {
            return;
        }

        const auto getNumberWithinRange = [](auto var, auto from, auto to) {
            return std::max(from, std::min(var, to));
        };

        auto finalChance = static_cast<int>(100.f / getNumberWithinRange(chance, 0.f, 100.f));

        auto shouldMutate = [&]() { return genRandom(0, finalChance) == 1; };

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
            color.r = static_cast<uint8_t>(genRandom(0, 255));
        }

        if (shouldMutate()) {
            color.g = static_cast<uint8_t>(genRandom(0, 255));
        }

        if (shouldMutate()) {
            color.b = static_cast<uint8_t>(genRandom(0, 255));
        }
    }

    circle &circle::operator=(const circle &rhs) noexcept {
        center = rhs.center;
        radius = rhs.radius;
        color = rhs.color;
        return *this;
    }
}