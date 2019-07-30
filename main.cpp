#include <iostream>
#include <vector>
#include "classes/shapes/circle.h"

using namespace vidga;

void checkIfContains(circle &circle1, coors c) {
    std::string containsStr = circle1.contains(c) ? "contains" : "does not contain";
    std::cout << "C " << containsStr << " " << c << std::endl;
}

int main() {
    circle circle1(20);

    auto contains = [&circle1](coors c) {
        checkIfContains(circle1, c);
    };

    contains({100, 0});
    contains({20, 0});
    contains({0, 20});
    contains({10, 10});
    contains({13, 13});
    contains({14, 14});
    contains({15, 15});
    return 0;
}