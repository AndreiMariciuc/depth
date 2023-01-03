//
// Created by andreimariciuc on 20.12.2022.
//

#include "disparityCalculator.h"
#include "utils.h"
#include <thread>

using namespace cv;
using namespace std;

DisparityCalculator::DisparityCalculator(cv::Mat_<img_t> &left, cv::Mat_<img_t> &right, int maxDisparity,
                                         int halfWindowX,
                                         int halfWindowY)
        : left(left),
          right(right),
          maxDisparity(maxDisparity),
          halfWindowX(halfWindowX),
          halfWindowY(halfWindowY) {
    assert(left.cols == right.cols && "Coloanele nu-s la fel!");
    assert(left.rows == right.cols && "Randurile nu-s la fel!");

    this->rows = this->left.rows;
    this->cols = this->left.cols;
}


void computeDisparityThread(Mat_<int> &disparity,
                            int ystart, int ystop,
                            DisparityCalculator &disp) {
    for (int y = ystart; y < ystop; y++) {
        for (int xl = disp.maxDisparity; xl < disp.cols; xl++) {
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