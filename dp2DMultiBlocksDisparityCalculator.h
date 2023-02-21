//
// Created by andreimariciuc on 21.12.2022.
//

#ifndef DEPTHESTIMATION_DP2DMULTIBLOCKSDISPARITYCALCULATOR_H
#define DEPTHESTIMATION_DP2DMULTIBLOCKSDISPARITYCALCULATOR_H

#include "disparityCalculator.h"

class DP2DMultiBlocksDisparityCalculator : public DisparityCalculator {
private:
    bool leftRightCheck = false;
    std::vector<std::pair<int, int>> blocks{
            {1,  32},
            {32, 1},
            {5, 5}
    };

    std::vector<std::vector<std::vector<ll>>> costsSumLeft;
    std::vector<std::vector<std::vector<ll>>> costsSumRight;

public:
    DP2DMultiBlocksDisparityCalculator(std::vector<std::vector<ll>> &left, std::vector<std::vector<ll>> &right, bool leftRightCheck = true,
                                       int maxDisparity = 150, int halfWindowX = 5, int halfWindowY = 5);

    DP2DMultiBlocksDisparityCalculator(std::vector<std::vector<ll>> &left, std::vector<std::vector<ll>> &right,bool leftRightCheck,
                                       std::vector<std::pair<int, int>> &blocks,
                                       int maxDisparity = 150, int halfWindowX = 5, int halfWindowY = 5);

private:

    int findDisparity(int y, int x) override;

    ll getMultiBlockCost(cv::Point3i p, const std::vector<std::vector<ll>> &sum);

    ll
    getBlockCost(const cv::Point3i &p, const std::vector<std::vector<ll>> &s, int curHalfWindowX, int curHalfWindowY);

    void costAggregation(
            const std::pair<std::vector<std::vector<std::vector<int>>>, std::vector<std::vector<std::vector<int>>>>& costs);

    std::vector<std::vector<ll>> partialSum(const std::vector<std::vector<int>> &costs);

    std::pair<std::vector<std::vector<std::vector<int>>>, std::vector<std::vector<std::vector<int>>>> computeCosts();
};

#endif //DEPTHESTIMATION_DP2DMULTIBLOCKSDISPARITYCALCULATOR_H
