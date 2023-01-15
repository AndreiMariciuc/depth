//
// Created by andreimariciuc on 20.12.2022.
//

#include "disparityCalculator.h"
#include "utils.h"
#include <thread>

using namespace cv;
using namespace std;

DisparityCalculator::DisparityCalculator(std::vector<std::vector<ll>> &left, std::vector<std::vector<ll>> &right,
                                         int maxDisparity,
                                         int halfWindowX,
                                         int halfWindowY)
        : left(left),
          right(right),
          maxDisparity(maxDisparity),
          halfWindowX(halfWindowX),
          halfWindowY(halfWindowY) {
    assert(left.cols == right.cols && "Coloanele nu-s la fel!");
    assert(left.rows == right.cols && "Randurile nu-s la fel!");

    this->rows = this->left.size();
    this->cols = this->left[0].size();
}


void computeDisparityThread(Mat_<int> &disparity,
                            int ystart, int ystop,
                            DisparityCalculator &disp) {
    for (int y = ystart; y < ystop; y++) {
        for (int xl = 0; xl < disp.cols; xl++) {
            disparity(y, xl) = disp.findDisparity(y, xl);
        }
    }
}

cv::Mat_<int> DisparityCalculator::computeDisparity(int nbOfThreads) {
    Mat_<int> disparity(rows, cols, 0);

    int dim = rows / nbOfThreads;
    vector<thread> threads(nbOfThreads);

    for (int i = 0; i < nbOfThreads; i++) {
        int ystart = i * dim;
        int ystop = i == nbOfThreads - 1 ? rows : (i + 1) * dim;

        threads[i] = thread(computeDisparityThread,
                            ref(disparity),
                            ystart,
                            ystop,
                            ref(*this));
    }

    for (auto &thread: threads) {
        thread.join();
    }

    return disparity;
}