//
// Created by andreimariciuc on 21.12.2022.
//

#include <thread>
#include "dp2DMultiBlocksDisparityCalculator.h"

using namespace std;
using namespace cv;

DP2DMultiBlocksDisparityCalculator::DP2DMultiBlocksDisparityCalculator(Mat_<img_t> &left,
                                                                       Mat_<img_t> &right,
                                                                       int maxDisparity, int halfWindowX,
                                                                       int halfWindowY)
        : DisparityCalculator(left, right,
                              maxDisparity,
                              halfWindowX,
                              halfWindowY) {
    this->costsSum = costAggregation(computeCosts());
}

DP2DMultiBlocksDisparityCalculator::DP2DMultiBlocksDisparityCalculator(Mat_<img_t> &left, Mat_<img_t> &right,
                                                                       std::vector<std::pair<int, int>> &blocks,
                                                                       int maxDisparity, int halfWindowX,
                                                                       int halfWindowY) : DisparityCalculator(left,
                                                                                                              right,
                                                                                                              maxDisparity,
                                                                                                              halfWindowX,
                                                                                                              halfWindowY) {
    this->blocks = blocks;
    this->costsSum = costAggregation(computeCosts());
}


vector<vector<ll>>
DP2DMultiBlocksDisparityCalculator::partialSum(const vector<vector<int>> &costs) {

    vector<vector<ll>> s(rows, vector<ll>(cols, 0));

    s[0][0] = costs[0][0];

    for (int j = 1; j < cols; j++) {
        s[0][j] = s[0][j - 1] + costs[0][j];
    }

    for (int i = 1; i < rows; i++) {
        s[i][0] = s[i - 1][0] + costs[i][0];
    }

    for (int i = 1; i < rows; i++)
        for (int j = 1; j < cols; j++)
            s[i][j] = s[i - 1][j] + s[i][j - 1] - s[i - 1][j - 1] + costs[i][j];

    return s;
}

vector<vector<vector<ll>>>
DP2DMultiBlocksDisparityCalculator::costAggregation(vector<vector<vector<int>>> costs) {
    vector<vector<vector<ll>>> sums(maxDisparity + 1);

    for (int d = 0; d <= maxDisparity; d++) {
        sums[d] = partialSum(costs[d]);
    }

    return sums;
}

vector<vector<vector<int>>>
DP2DMultiBlocksDisparityCalculator::computeCosts() {
    vector<vector<vector<int>>> costs(maxDisparity + 1,
                                      vector<vector<int >>(rows, vector<int>(cols, 0)));

    for (int d = 0; d <= maxDisparity; d++) {
        for (int y = 0; y < rows; y++) {
            for (int x = maxDisparity; x < cols; x++) {
                costs[d][y][x] = 8 - __builtin_popcount(left(y, x) ^ right(y, x - d));
            }
        }
    }

    return costs;
}

ll DP2DMultiBlocksDisparityCalculator::getBlockCost(const Point3i &p, const vector<vector<ll>> &s, int curHalfWindowX, int curHalfWindowY) {
    return s[min(p.y + curHalfWindowY, rows - 1)][min(p.x + curHalfWindowX, cols - 1)]
           - s[max(p.y - curHalfWindowY, 0)][min(p.x + curHalfWindowX, cols - 1)]
           - s[min(p.y + curHalfWindowY, rows - 1)][max(p.x - curHalfWindowX, 0)]
           + s[max(p.y - curHalfWindowY, 0)][max(p.x - curHalfWindowX, 0)];
}

ll DP2DMultiBlocksDisparityCalculator::getMultiBlockCost(Point3i p, const vector<vector<ll>> &sum) {
    ll score = 1;

    for (int i = 0; i < blocks.size(); i++) {
        halfWindowX = blocks[i].first;
        halfWindowY = blocks[i].second;
        if (i == 1) {
            ll curScore = getBlockCost(p, sum, blocks[i].first, blocks[i].second);
            score = max(score, curScore);
        } else {
            score *= getBlockCost(p, sum, blocks[i].first, blocks[i].second);
        }
    }

    return score;
}

int DP2DMultiBlocksDisparityCalculator::findDisparity(int y, int xl) {
    ll bestScore = getMultiBlockCost(Point3i(xl, y, 0), costsSum[0]);
    int bestDisparity = 0;

    for (int d = 1; d <= maxDisparity; d++) {
        ll score = getMultiBlockCost(Point3i(xl, y, d), costsSum[d]);
        if (score > bestScore) {
            bestScore = score;
            bestDisparity = d;
        }
    }

    return bestDisparity;
}




