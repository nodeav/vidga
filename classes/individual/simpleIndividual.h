//
// Created by Nadav Eidelstein on 03/08/2019.
//

#ifndef VIDGA_SIMPLECHROMOSOME_H
#define VIDGA_SIMPLECHROMOSOME_H

#include <vector>
#include <array>
#include <memory>
#include <limits>
#include <iostream>

#include "opencv2/core/mat.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"


#include "../../interfaces/individual/individual.h"
#include "../shapes/circle.h"

namespace vidga {
    class simpleIndividual : individual {
    public:
        simpleIndividual(size_t size, ucoor_t sideLengthMin, ucoor_t sideLengthMax, ucoor_t xMax, ucoor_t yMax);

//        std::vector<circle> &getShapes() const override;

        std::vector<circle> &getShapes() override;

        void draw(float3 *canvas, float **map) const;

        std::shared_ptr<simpleIndividual> randMerge(std::shared_ptr<simpleIndividual> src, ucoor_t sideLengthMin,
                                                    ucoor_t sideLengthMax, ucoor_t xMax, ucoor_t yMax);

        void calcAndSetScore(float3 *target, float3 *canvas, float **circlesMap);

        float getScore() const;

    private:
        std::vector<circle> circles;
        unsigned width, height;
        float score = 4;

        ucoor_t minMapRadius;


        bool targetCopied{false};
        float *targetCPU, *canvasCPU;
        cv::Mat targetMat, canvasMat;
    };
}

#endif //VIDGA_SIMPLECHROMOSOME_H
