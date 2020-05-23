#include <vector>
#include <memory>

#include "../population/simplePopulation.h"

#include "SerDes.h"

namespace vidga {
    std::vector<uint8_t> SerDes::serialize(std::shared_ptr<simplePopulation> population) {
        return std::vector<uint8_t>();
    }

    std::shared_ptr<simplePopulation> SerDes::deserialize(std::vector<uint8_t> arr) {
        return std::make_shared<simplePopulation>(1, 1, 1, 1);
    }
}