//
// Created by Nadav Eidelstein on 27/07/2019.
//
#ifndef VIDGA_SHAPE_H
#define VIDGA_SHAPE_H

#include <cmath>
#include <ostream>
#include "opencv2/core/types.hpp"
#include <random>
#include <vector_types.h>

namespace vidga {
    typedef unsigned int ucoor_t;
    typedef int coor_t;

    static std::random_device randomDevice;
    static std::mt19937 numberGenerator(randomDevice());

    static float genRandom(float from, float to) {
        return std::uniform_real_distribution<float>(from, to)(numberGenerator);
    }

//    typedef struct bgr_color_t {
//        float r, g, b, a;
//        explicit operator cv::Scalar() const {
//            return cv::Scalar(b, g, r, a);
//        }
//        friend std::ostream& operator<<(std::ostream& stream, const bgr_color_t& c) {
//            stream << "(" << std::to_string(c.r)
//                   << ", " << std::to_string(c.g)
//                   << ", " << std::to_string(c.b)
//                   << ", " << std::to_string(c.a)
//                   << ")";
//            return stream;
//        }
//    } bgr_color_t;
    typedef float4 bgr_color_t;

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

            return sqrtf(xDiffSquared + yDiffSquared);
        }

        explicit operator cv::Point() const {
            return {static_cast<int>(x), static_cast<int>(y)};
        }

    } coors;

    class shape {
    public:
        explicit shape(coors center_);
        virtual bool contains(coors c) const = 0;
        virtual void mutate(float chance, ucoor_t xMax, ucoor_t yMax, ucoor_t sizeMin, ucoor_t sizeMax) = 0;

        // Getters
        virtual coors getCenter() const;
        virtual ucoor_t getWidth() const = 0;
        virtual ucoor_t getHeight() const = 0;
        virtual bgr_color_t getColor() const = 0;

        // Setters
        void setCenter(coors c);
        virtual void setWidth(ucoor_t newWidth) = 0;
        virtual void setHeight(ucoor_t newHeight) = 0;
        virtual void setColor(bgr_color_t newColor) = 0;
        virtual void setRandomEverything(ucoor_t sideLengthMin, ucoor_t sideLengthMax, ucoor_t xMax, ucoor_t yMax) = 0;

    private:
        coors center;
        ucoor_t width, height;
        bgr_color_t color;
    };

}
#endif //VIDGA_SHAPE_H
