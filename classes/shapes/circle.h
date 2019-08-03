//
// Created by Nadav Eidelstein on 30/07/2019.
//

#ifndef VIDGA_CIRCLE_H
#define VIDGA_CIRCLE_H

#include "../../interfaces/shape/shape.h"

namespace vidga {
    class circle : public shape {
    public:
        circle(coors center_, ucoor_t radius_);
        explicit circle(ucoor_t radius_);

        bool contains(coors c) const override;
        ucoor_t getWidth() const override;
        ucoor_t getHeight() const override;

        void setWidth(ucoor_t newWidth) override;
        void setHeight(ucoor_t newHeight) override;
        void setRandom(ucoor_t sideLengthMin, ucoor_t sideLengthMax, ucoor_t xMax, ucoor_t yMax) override;
    private:
        coors center;
        ucoor_t radius;
    };
}


#endif //VIDGA_CIRCLE_H
