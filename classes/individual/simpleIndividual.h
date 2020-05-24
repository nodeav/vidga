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
        simpleIndividual(size_t size, ucoor_t sideLengthMin, ucoor_t sideLengthMax, ucoor_t xMax,
                         ucoor_t yMax, bool setRandom = false);
        const std::vector<std::shared_ptr<circle>>& getShapes() const override;
        std::vector<std::shared_ptr<circle>>& getShapesMut() override;
        void draw(cv::Mat& canvas) const;
        std::shared_ptr<simpleIndividual> randMerge(std::shared_ptr<simpleIndividual> src, ucoor_t sideLengthMin,
                                     ucoor_t sideLengthMax, ucoor_t xMax, ucoor_t yMax);
        void calcAndSetScore(cv::Mat& target, cv::Mat& canvas, cv::Mat& dst);
        float getScore() const;

    private:
        void sortShapes();

        std::vector<std::shared_ptr<circle>> shapes;
        float score = 4;

    };
}

#endif //VIDGA_SIMPLECHROMOSOME_H
