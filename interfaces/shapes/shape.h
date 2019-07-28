//
// Created by Nadav Eidelstein on 27/07/2019.
//

#ifndef VIDGA_SHAPE_H
#define VIDGA_SHAPE_H

namespace vidga {

    typedef struct coors {
        unsigned int x;
        unsigned int y;
    } coors;

    template <class T>
    class shape {
    public:
        // Getters
        virtual coors getCenter() const = 0;
        virtual unsigned int getWidth() const = 0;
        virtual unsigned int getHeight() const = 0;

        // Setters
        virtual void setCenter(coors c) = 0;
        virtual void setWidth(unsigned int newWidth) = 0;
        virtual void setHeight(unsigned int newHeight) = 0;
    private:
        coors center;
        unsigned int width, height;
    };
}
#endif //VIDGA_SHAPE_H
