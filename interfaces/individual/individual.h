//
// Created by Nadav Eidelstein on 27/07/2019.
//

#ifndef VIDGA_CHROMOSOME_H
#define VIDGA_CHROMOSOME_H

#include <vector>
#include "../shape/shape.h"

namespace vidga {

    class individual {
    public:
        virtual const std::vector<std::shared_ptr<shape>>& getShapes() const = 0;
        virtual std::vector<std::shared_ptr<shape>>& getShapesMut() = 0;
    private:
        std::vector<std::shared_ptr<shape>> shapes;
    };

}

#endif //VIDGA_CHROMOSOME_H
