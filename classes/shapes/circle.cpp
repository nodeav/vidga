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
        : radius(radius_)
    {
        // TODO: use delegated constructor (values overridden somehow?)
        setCenter(center_);
    }

    circle::circle(ucoor_t radius_)
        : radius(radius_)
    {
        std::cout << "radius is " << radius << " and coors are {" << center.x << ", " << center.y << "}" << std::endl;

    }
}