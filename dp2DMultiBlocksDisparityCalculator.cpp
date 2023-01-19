//
// Created by andreimariciuc on 21.12.2022.
//

#include <thread>
#include "dp2DMultiBlocksDisparityCalculator.h"

using namespace std;
using namespace cv;

DP2DMultiBlocksDisparityCalculator::DP2DMultiBlocksDisparityCalculator(std::vector<std::vector<ll>> &left,
                                                                       std::vector<std::vector<ll>> &right,
                                                                       bool leftRightCheck,
                                                                       int maxDisparity, int halfWindowX,
                                                                       int halfWindowY)
        : DisparityCalculator(left, right,
                              maxDisparity,
                              halfWindowX,
                              halfWindowY) {
    this->leftRightCheck = leftRightCheck;
    this->costAggregation(computeCosts());
}

DP2DMultiBlocksDisparityCalculator::DP2DMultiBlocksDisparityCalculator(std::vector<std::vector<ll>> &left,
                                                                       std::vector<std::vector<ll>> &right,
                                                                       bool leftRightCheck,
                                                                       std::vector<std::pair<int, int>> &blocks,
                                                                       int maxDisparity, int halfWindowX,
                                                                       int halfWindowY) : DisparityCalculator(left,
                                                                                                              right,
                                                                                                              maxDisparity,
                                                                                                              halfWindowX,
                                                                                                              halfWindowY) {
    this->blocks = blocks;
    this->leftRightCheck = leftRightCheck;
    this->costAggregation(computeCosts());
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

void
DP2DMultiBlocksDisparityCalculator::costAggregation(
        const pair<vector<vector<vector<int>>>, vector<vector<vector<int>>>> &costs) {
    this->costsSumLeft.resize(maxDisparity + 1);
    if (leftRightCheck)
        this->costsSumRight.resize(maxDisparity + 1);

    auto costsLeft = costs.first;
    auto costsRight = costs.second;

    for (int d = 0; d <= maxDisparity; d++) {
        this->costsSumLeft[d] = partialSum(costsLeft[d]);
        if (leftRightCheck)
            this->costsSumRight[d] = partialSum(costsRight[d]);
    }
}

pair<vector<vector<vector<int>>>, vector<vector<vector<int>>>>
DP2DMultiBlocksDisparityCalculator::computeCosts() {
    vector<vector<vector<int>>> costsLeft(maxDisparity + 1,
                                          vector<vector<int >>(rows, vector<int>(cols, 0)));

    vector<vector<vector<int>>> costsRight(maxDisparity + 1,
                                           vector<vector<int >>(rows, vector<int>(cols, 0)));


    for (int d = 0; d <= maxDisparity; d++) {
        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                if (x >= maxDisparity) {
                    ll v = left[y][x] ^ right[y][x - d];
                    int h1 = __builtin_popcount(v);
                    int h2 = __builtin_popcount(v >> 32);
                    costsLeft[d][y][x] = 63 - (h1 + h2);
                }

                if (leftRightCheck && x <= cols - maxDisparity) {
                    ll v = left[y][x + d] ^ right[y][x];
                    int h1 = __builtin_popcount(v);
                    int h2 = __builtin_popcount(v >> 32);
                    costsRight[d][y][x] = 63 - (h1 + h2);
                }
            }
        }
    }

    return {costsLeft, costsRight};
}

ll DP2DMultiBlocksDisparityCalculator::getBlockCost(const Point3i &p, const vector<vector<ll>> &s, int curHalfWindowX,
                                                    int curHalfWindowY) {
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

int DP2DMultiBlocksDisparityCalculator::findDisparity(int y, int x) {
    ll bestLeftScore = getMultiBlockCost(Point3i(x, y, 0), costsSumLeft[0]);
    ll bestRightScore;
    if (leftRightCheck)
        bestRightScore = getMultiBlockCost(Point3i(x, y, 0), costsSumRight[0]);
    int bestDisparityLeft = 0;
    int bestDisparityRight = 0;

    for (int d = 1; d <= maxDisparity; d++) {

        ll score = getMultiBlockCost(Point3i(x, y, d), costsSumLeft[d]);
        if (score > bestLeftScore) {
            bestLeftScore = score;
            bestDisparityLeft = d;
        }

        if (leftRightCheck) {
            score = getMultiBlockCost(Point3i(x, y, d), costsSumRight[d]);
            if (score > bestRightScore) {
                bestRightScore = score;
                bestDisparityRight = d;
            }
        }

    }

//    return bestDisparityLeft == bestDisparityRight ? bestDisparityLeft : 0;
//    return bestDisparityRight;
    if (leftRightCheck) {
        if (x < maxDisparity) return bestDisparityRight;
        if (x > cols - maxDisparity) return bestDisparityLeft;

        return abs(bestDisparityRight - bestDisparityLeft) < 10 ? (bestDisparityRight + bestDisparityLeft) / 2 : 0;
    }

    return bestDisparityLeft;
}




