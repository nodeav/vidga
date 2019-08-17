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
#include <opencv2/quality/qualitymse.hpp>


#include "../../interfaces/individual/individual.h"
#include "../shapes/circle.h"

namespace vidga {
    class simpleIndividual : individual {
    public:
        simpleIndividual(size_t size, ucoor_t sideLengthMin, ucoor_t sideLengthMax, ucoor_t xMax, ucoor_t yMax);
        const std::vector<std::unique_ptr<shape>>& getShapes() const override;
        std::vector<std::unique_ptr<shape>>& getShapesMut() override;
        void draw(cv::Mat& canvas) const;
        void mutRandMerge(simpleIndividual &src);
        void calcAndSetScore(cv::Mat& target, cv::Mat& canvas);
        float getScore() const;

    private:
        std::vector<std::unique_ptr<shape>> shapes;
        float score;

    };
}

#endif //VIDGA_SIMPLECHROMOSOME_H
