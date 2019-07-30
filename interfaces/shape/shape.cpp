//
// Created by Nadav Eidelstein on 27/07/2019.
//

#include "shape.h"
namespace vidga {
    void shape::setCenter(coors c) {
        center = c;
    }

    coors shape::getCenter() const {
        return center;
    }

    shape::shape(coors center_) : center(center_) {

    }
}