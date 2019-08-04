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
        virtual const std::vector<std::unique_ptr<shape>>& getShapes() const = 0;
        virtual std::vector<std::unique_ptr<shape>>& getShapesMut() = 0;
    private:
        std::vector<std::unique_ptr<shape>> shapes;
    };

}

#endif //VIDGA_CHROMOSOME_H
