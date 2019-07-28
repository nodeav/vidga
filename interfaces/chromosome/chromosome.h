//
// Created by Nadav Eidelstein on 27/07/2019.
//

#ifndef VIDGA_CHROMOSOME_H
#define VIDGA_CHROMOSOME_H

#include <array>
#include "../shapes/shape.h"

namespace vidga {

    class chromosome {
    public:
        // Move ctor
        virtual chromosome(std::vector<shape>&& shapes) = 0;
        auto getChromosomes() const = 0;
        auto getMutChromosomes() = 0;

    private:
        std::vector<shape> shapes;
    };

}

#endif //VIDGA_CHROMOSOME_H
