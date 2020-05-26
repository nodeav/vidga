//
// Created by Nadav Eidelstein on 03/08/2019.
//
#include "simpleIndividual.h"
#include <mutex>
#include "util.h"

cv::Scalar getMSSIM( const cv::Mat& i1, const cv::Mat& i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d     = CV_32F;

    cv::Mat I1, I2;
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    cv::Mat I2_2   = I2.mul(I2);        // I2^2
    cv::Mat I1_2   = I1.mul(I1);        // I1^2
    cv::Mat I1_I2  = I1.mul(I2);        // I1 * I2

    /*************************** END INITS **********************************/

    cv::Mat mu1, mu2;   // PRELIMINARY COMPUTING
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2   =   mu1.mul(mu1);
    cv::Mat mu2_2   =   mu2.mul(mu2);
    cv::Mat mu1_mu2 =   mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;

    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    cv::Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    cv::Mat ssim_map;
    cv::divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

    cv::Scalar mssim = cv::mean( ssim_map ); // mssim = average of ssim map
    return mssim;
}

namespace vidga {
    std::vector<std::shared_ptr<circle>> &simpleIndividual::getShapesMut() {
        return shapes;
    }

    const std::vector<std::shared_ptr<circle>> &simpleIndividual::getShapes() const {
        return shapes;
    }

    simpleIndividual::simpleIndividual(size_t size, ucoor_t sideLengthMin, ucoor_t sideLengthMax, ucoor_t xMax,
                                       ucoor_t yMax, bool setRandom) {
        shapes.reserve(size);
        for (auto i = 0; i < size; i++) {
            auto c = std::make_shared<circle>();
            if (setRandom) {
                c->setRandomEverything(sideLengthMin, sideLengthMax, xMax, yMax);
            }
            shapes.push_back(c);
        }
        sortShapes();
    }

    void simpleIndividual::draw(cv::Mat &canvas) const {
        int i = 0;
        for (auto const &circle : shapes) {
            if (circle == nullptr) {
                std::cout << "circle #" << i++ << " is null!" << std::endl;
                continue;
            }
            const auto pt = cv::Point(circle->getCenter());
            cv::circle(canvas, pt, circle->getWidth(), cv::Scalar(circle->getColor()), -1);
        }
    }

    const auto getBit = [](int bits, int index) {
        return (bits >> index) & 1;
    };

    const auto genRandomInt() {
        return genRandom(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    }

    std::shared_ptr<simpleIndividual>
    simpleIndividual::randMerge(std::shared_ptr<simpleIndividual> src, ucoor_t sideLengthMin,
                                ucoor_t sideLengthMax, ucoor_t xMax, ucoor_t yMax) {
        auto dst = std::make_shared<simpleIndividual>(0, sideLengthMin, sideLengthMax, xMax, yMax);
        auto &dstShapes = dst->getShapesMut();

        auto &srcShapes = src->getShapesMut();
        dstShapes.reserve(src->getShapes().size());

        // We only need 1 bit of randomness per decision
        // const auto bitsPerInt = sizeof(int) * 8;
        // const auto intsOfRandomness = static_cast<int>(dstShapes.size() / bitsPerInt + 1);

        // for (auto i = 0; i < intsOfRandomness; i++) {
        //     auto oneInt = genRandomInt();
        //     auto idx = i * bitsPerInt;
        //     for (int j = 0; j < bitsPerInt && idx < dstShapes.size(); j++, idx++) {
        //         std::shared_ptr<circle> ptr;
        //         if (getBit(oneInt, j)) {
        //             ptr = srcShapes[idx];
        //         } else {
        //             ptr = shapes[idx];
        //         }
        //         *dstShapes[idx] = *ptr;
        //         dstShapes[idx]->mutate(0.1, xMax, yMax, sideLengthMin, sideLengthMax);

        //         if (genRandom(0, 50) == 1) {
        //             auto idx1 = genRandom(0, static_cast<int>(dstShapes.size() - 1));
        //             auto idx2 = genRandom(0, static_cast<int>(dstShapes.size() - 1));
        //             std::iter_swap(dstShapes.begin() + idx1, dstShapes.begin() + idx2);
        //         }
        //     }
        // }

        for (int i=0; i < srcShapes.size(); i++) {
            if (genRandom(0, 100) < 50) {
                dstShapes.push_back(std::make_shared<circle>(*shapes[i]));
            } else {
                dstShapes.push_back(std::make_shared<circle>(*srcShapes[i]));
            }

            dstShapes[i]->mutate(0.1, xMax, yMax, sideLengthMin, sideLengthMax);
        }

        dst->sortShapes();
        return dst;
    }

    void simpleIndividual::calcAndSetScore(cv::Mat &target, cv::Mat &canvas, cv::Mat &dst) {
        draw(canvas);
         cv::absdiff(target, canvas, dst);
         cv::Scalar newScore = cv::sum(dst);
         score = static_cast<float>((newScore.val[0] + newScore.val[1] + newScore.val[2]) /
                                    (canvas.total() * canvas.channels()));
//
//        cv::Scalar mssim = getMSSIM(target, canvas);
//        score = 1.0f - static_cast<float>((mssim.val[0] + mssim.val[1] + mssim.val[2]) / 3);
    }

    float simpleIndividual::getScore() const {
        return score;
    }

    void simpleIndividual::sortShapes() {
        std::sort(shapes.begin(), shapes.end(), [](std::shared_ptr<circle> c1, std::shared_ptr<circle> c2){ return c1->getRadius() > c2->getRadius();});
    }
}
