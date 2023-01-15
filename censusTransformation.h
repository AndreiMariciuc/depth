//
// Created by andreimariciuc on 09.11.2022.
//

#ifndef DEPTHESTIMATION_CENSUSTRANSFORMATION_H
#define DEPTHESTIMATION_CENSUSTRANSFORMATION_H

#include <opencv2/opencv.hpp>
#include "utils.h"

using namespace cv;


std::vector<std::vector<ll>> censusTr(const Mat_<uchar> &img, int nbOfThreads = 8);

#endif //DEPTHESTIMATION_CENSUSTRANSFORMATION_H
