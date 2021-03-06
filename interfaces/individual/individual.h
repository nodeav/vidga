//
// Created by Nadav Eidelstein on 27/07/2019.
//

#ifndef VIDGA_CHROMOSOME_H
#define VIDGA_CHROMOSOME_H

#include <vector>
//#include "../shape/shape.h"
#include "../../classes/shapes/circle.h"

namespace vidga {

    class individual {
    public:
        virtual std::vector<circle> & getShapes() = 0;
//        virtual std::vector<circle> & getShapesMut() = 0;
    private:
        std::vector<std::shared_ptr<circle>> shapes;
    };

}

#endif //VIDGA_CHROMOSOME_H
