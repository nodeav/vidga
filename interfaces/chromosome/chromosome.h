//
// Created by Nadav Eidelstein on 27/07/2019.
//

#ifndef VIDGA_CHROMOSOME_H
#define VIDGA_CHROMOSOME_H

#include <vector>
#include "../shape/shape.h"

namespace vidga {

    class chromosome {
    public:
        virtual std::vector<shape> getShapes() const = 0;
        virtual std::vector<shape> getShapesMut() = 0;

    private:
        std::vector<shape> shapes;
    };

}

#endif //VIDGA_CHROMOSOME_H
