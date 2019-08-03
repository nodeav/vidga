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

    void circle::setHeight(ucoor_t newHeight) {
        radius = newHeight;
    }

    void circle::setWidth(ucoor_t newWidth) {
        radius = newWidth;
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
    }
}