//
// Created by andreimariciuc on 09.11.2022.
//

#ifndef DEPTHESTIMATION_CENSUSTRANSFORMATION_H
#define DEPTHESTIMATION_CENSUSTRANSFORMATION_H

#include <opencv2/opencv.hpp>

using namespace cv;

Mat_<unsigned short> censusTr(const Mat_<uchar> &img, int nbOfThreads = 8);

#endif //DEPTHESTIMATION_CENSUSTRANSFORMATION_H
