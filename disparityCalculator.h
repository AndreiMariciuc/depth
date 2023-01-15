//
// Created by andreimariciuc on 20.12.2022.
//

#ifndef DEPTHESTIMATION_DISPARITYCALCULATOR_H
#define DEPTHESTIMATION_DISPARITYCALCULATOR_H

#include <opencv2/opencv.hpp>
#include "utils.h"


class DisparityCalculator {
protected:
    int maxDisparity;
    int halfWindowX, halfWindowY;
    int rows, cols;
    std::vector<std::vector<ll>> left, right;

    virtual int findDisparity(int y, int xl) = 0;

public:
    DisparityCalculator(std::vector<std::vector<ll>> &left, std::vector<std::vector<ll>> &right,
                        int maxDisparity = 150, int halfWindowX = 5, int halfWindowY = 5);

    cv::Mat_<int> computeDisparity(int nbOfThreads = 8);

private:
    friend void computeDisparityThread(cv::Mat_<int> &disparity,
                                       int ystart, int ystop,
                                       DisparityCalculator &disp);

};

#endif //DEPTHESTIMATION_DISPARITYCALCULATOR_H
