//
// Created by Nadav Eidelstein on 30/07/2019.
//

#ifndef VIDGA_CIRCLE_H
#define VIDGA_CIRCLE_H

#include <cuda_runtime.h>
#include "../../interfaces/shape/shape.h"

namespace vidga {
    class circle {
    public:
        circle();
        circle(coors center_, ucoor_t radius_);
        explicit circle(ucoor_t radius_);
        void mutate(float chance, ucoor_t xMax, ucoor_t yMax, ucoor_t sizeMin, ucoor_t sizeMax);
//        circle& operator=(const circle &rhs) noexcept;

        bool contains(coors c) const;
        ucoor_t getWidth() const;
        ucoor_t getHeight() const;
        coors getCenter() const;
        bgr_color_t getColor() const;

        void setWidth(ucoor_t newWidth);
        void setHeight(ucoor_t newHeight);
        void setColor(bgr_color_t newColor);
        void setRandomEverything(ucoor_t sideLengthMin, ucoor_t sideLengthMax, ucoor_t xMax, ucoor_t yMax);

//    private:
        coors center;
        ucoor_t radius;
        bgr_color_t color;
    };
}


#endif //VIDGA_CIRCLE_H
