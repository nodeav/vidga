//
// Created by Nadav Eidelstein on 27/07/2019.
//
#ifndef VIDGA_SHAPE_H
#define VIDGA_SHAPE_H

#include <cmath>
#include <ostream>
#include "opencv2/core/types.hpp"
#include <random>

namespace vidga {
    typedef unsigned int ucoor_t;
    typedef int coor_t;

    static std::random_device randomDevice;
    static std::mt19937 numberGenerator(randomDevice());

    static int genRandom(int from, int to) {
        return std::uniform_int_distribution<int>(from, to)(numberGenerator);
    }

    typedef struct coors {
        ucoor_t x;
        ucoor_t y;


        static coors generateRandom(ucoor_t xMax, ucoor_t yMax) {
            auto x = static_cast<ucoor_t>(genRandom(0, xMax));
            auto y = static_cast<ucoor_t>(genRandom(0, yMax));
            return {x, y};
        }

        // Enabled printing
        friend std::ostream& operator<<(std::ostream& stream, const coors& c) {
            stream << "{" << c.x << ", " << c.y << "}";
            return stream;
        }

        float distanceFrom(coors c) const {
            auto getSquaredDiff = [](auto x1, auto x2) {
                return (x2 - x1) * (x2 - x1);
            };
            coor_t xDiffSquared = getSquaredDiff(x, c.x),
                   yDiffSquared = getSquaredDiff(y, c.y);

            return std::sqrtf(xDiffSquared + yDiffSquared);
        }

        explicit operator cv::Point() const {
            return {static_cast<int>(x), static_cast<int>(y)};
        }

    } coors;

    class shape {
    public:
        explicit shape(coors center_);
        virtual bool contains(coors c) const = 0;
        // Getters
        coors getCenter() const;
        virtual ucoor_t getWidth() const = 0;
        virtual ucoor_t getHeight() const = 0;

        // Setters
        void setCenter(coors c);
        virtual void setWidth(ucoor_t newWidth) = 0;
        virtual void setHeight(ucoor_t newHeight) = 0;
        virtual void setRandom(ucoor_t sideLengthMin, ucoor_t sideLengthMax, ucoor_t xMax, ucoor_t yMax) = 0;

    private:
        coors center;
        ucoor_t width, height;
    };

}
#endif //VIDGA_SHAPE_H
