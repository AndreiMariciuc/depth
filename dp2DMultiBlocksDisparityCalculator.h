//
// Created by andreimariciuc on 21.12.2022.
//

#ifndef DEPTHESTIMATION_DP2DMULTIBLOCKSDISPARITYCALCULATOR_H
#define DEPTHESTIMATION_DP2DMULTIBLOCKSDISPARITYCALCULATOR_H

#include "disparityCalculator.h"

class DP2DMultiBlocksDisparityCalculator : public DisparityCalculator {
private:
    std::vector<std::pair<int, int>> blocks{
            {1,  32},
            {32, 1},
            {6,  6},
//            {4,  4},
    };

    std::vector<std::vector<std::vector<ll>>> costsSum;

public:
    DP2DMultiBlocksDisparityCalculator(cv::Mat_<img_t> &left, cv::Mat_<img_t> &right,
                                       int maxDisparity = 150, int halfWindowX = 5, int halfWindowY = 5);

    DP2DMultiBlocksDisparityCalculator(cv::Mat_<img_t> &left, cv::Mat_<img_t> &right,
                                       std::vector<std::pair<int, int>> &blocks,
                                       int maxDisparity = 150, int halfWindowX = 5, int halfWindowY = 5);

private:

    int findDisparity(int y, int xl) override;

    ll getMultiBlockCost(cv::Point3i p, const std::vector<std::vector<ll>> &sum);

    ll getBlockCost(const cv::Point3i &p, const std::vector<std::vector<ll>> &s, int curHalfWindowX, int curHalfWindowY);

    std::vector<std::vector<std::vector<ll>>> costAggregation(std::vector<std::vector<std::vector<int>>> costs);

    std::vector<std::vector<ll>> partialSum(const std::vector<std::vector<int>> &costs);

    std::vector<std::vector<std::vector<int>>> computeCosts();
};

#endif //DEPTHESTIMATION_DP2DMULTIBLOCKSDISPARITYCALCULATOR_H
