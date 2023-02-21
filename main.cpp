#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "utils.h"
#include "censusTransformation.h"
#include "dp2DMultiBlocksDisparityCalculator.h"

using namespace std;
using namespace cv;

const string LEFT_PATH = "../kitti/training/image_2";
const string RIGHT_PATH = "../kitti/training/image_3";

vector<vector<ll>> leftCens, rightCens;

inline Mat_<uchar> scaleImg(const Mat_<int> &m) {
    auto maxim = *max_element(m.begin(), m.end());
    return 255 * m / maxim;
}

inline Mat_<uchar> scaleImg(const Mat_<float> &m) {
    auto maxim = *max_element(m.begin(), m.end());
    return 255 * m / maxim;
}

Mat_<int> compute2dHist(Mat_<int> du, Mat_<int> dv) {
    Mat_<int> hist(256, 256, 0);
//    cout << du;

    for (int i = 0; i < du.rows; i++) {
        for (int j = 0; j < dv.cols; j++) {
            hist(dv(i, j), du(i, j)) = hist(dv(i, j), du(i, j)) + 1;
//            cout << int(du(i, j)) << " " << int(dv(i, j)) << "=" << int(hist(du(i, j), dv(i, j))) << "\n";
        }
    }

    return hist;
}

Mat_<float> convolution(const Mat_<uchar> &img, const Mat_<float> &k) {
    Mat_<float> newImg(img.rows, img.cols);
    pair<int, int> kCenter(k.rows / 2, k.cols / 2);
    pair<int, int> limit(k.rows % 2 + kCenter.first, k.cols % 2 + kCenter.second);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            float s = 0;
            for (int ki = -kCenter.first; ki < limit.first; ki++)
                for (int kj = -kCenter.second; kj < limit.second; kj++)
                    if (isInside(ki + i, kj + j, img.rows, img.cols))
                        s += float(img(ki + i, kj + j)) * k(ki + kCenter.first, kj + kCenter.second);

            newImg(i, j) = s;
        }
    }

    return newImg;
}

Mat_<uchar> normalization(const Mat_<float> &img, const Mat_<float> &k) {
    float s_plus = 0, s_minus = 0;

    for (int i = 0; i < k.rows; i++) {
        for (int j = 0; j < k.cols; j++) {
            if (k(i, j) > 0) s_plus += k(i, j);
            else
                s_minus += k(i, j);
        }
    }

//    return abs(img) / max(s_plus, -s_minus);
    return 128 + img / (2 * max(s_plus, -s_minus));
}


Mat_<float> convertToLg(const Mat_<int> &h) {
    Mat_<float> result(h.rows, h.cols);
    for (int i = 0; i < h.rows; i++) {
        for (int j = 0; j < h.cols; j++) {
            result(i, j) = log(h(i, j) + 1);
        }
    }
    return result;
}

void createVideo() {
    VideoWriter video("disp_v5.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 5, Size(1242, 375 + 375 + 375));

    char fileName[20];
    Mat toWrite;
    for (int sampleIdx = 0;; sampleIdx++) {
        sprintf(fileName, "/%06d", sampleIdx);

        Mat left = imread(LEFT_PATH + fileName + "_10.png", IMREAD_GRAYSCALE);
        if (left.cols == 0) break;
        Mat right = imread(RIGHT_PATH + fileName + "_10.png", IMREAD_GRAYSCALE);

        leftCens = censusTr(left);
        rightCens = censusTr(right);

//        auto dispCalc = DP2DMultiBlocksDisparityCalculator(leftCens, rightCens);
//        Mat_<int> disparity = dispCalc.computeDisparity();
//        auto disp = scaleImg(disparity);
//        imshow("disp", disp);
//        waitKey(0);
        cvtColor(scaleImg(DP2DMultiBlocksDisparityCalculator(leftCens, rightCens).computeDisparity()), toWrite,
                 COLOR_GRAY2BGR);
        Mat toWriteLeft;
        cvtColor(left, toWriteLeft,
                 COLOR_GRAY2BGR);
        Mat heatmap;
        applyColorMap(toWrite, heatmap, COLORMAP_JET);
        vector<Mat> matrices1 = {
                toWriteLeft,
                toWrite,
                heatmap
        };
        Mat R1;
        vconcat(matrices1, R1);
        video << R1;

        left = imread(LEFT_PATH + fileName + "_11.png", IMREAD_GRAYSCALE);
        right = imread(RIGHT_PATH + fileName + "_11.png", IMREAD_GRAYSCALE);

        leftCens = censusTr(left);
        rightCens = censusTr(right);

        cvtColor(scaleImg(DP2DMultiBlocksDisparityCalculator(leftCens, rightCens).computeDisparity()), toWrite,
                 COLOR_GRAY2BGR);
        applyColorMap(toWrite, heatmap, COLORMAP_JET);
        cvtColor(left, toWriteLeft,
                 COLOR_GRAY2BGR);
        vector<Mat> matrices = {
                toWriteLeft,
                toWrite,
                heatmap
        };
        Mat R;
        vconcat(matrices, R);
        video << R;
        cout << sampleIdx << " done" << "\n";
    }
}


Mat_<uchar> shearAndScale(const Mat_<uchar> &right, float g = 2, float s = 1) {
    Mat_<uchar> ssRight(right.rows, right.cols, (uchar) 0);

    for (int i = 0; i < right.rows; i++) {
        for (int j = 0; j < right.cols; j++) {
            int jj = s * (j + g * i);
            if (jj < right.cols && jj >= 0)
                ssRight(i, jj) = right(i, j);
        }
    }

    return ssRight;
}

int main() {
#ifdef CREATE_VIDEO
    createVideo();
    return 0;
#endif

    Mat_<uchar> left = imread(LEFT_PATH + "/000001_10.png", IMREAD_GRAYSCALE), right = imread(
            RIGHT_PATH + "/000001_10.png", IMREAD_GRAYSCALE);
//    Mat_<uchar> left = imread("../sampledata/im00.png", IMREAD_GRAYSCALE), right = imread("../sampledata/im11.png", IMREAD_GRAYSCALE);


#ifdef SHEAR_SCALE
    auto ssRight = shearAndScale(right, 2, 1.5);
    imshow("right", right);
    imshow("ssRight", ssRight);
    waitKey(0);
    return 0;
#endif

    leftCens = censusTr(left);
    rightCens = censusTr(right);

    auto disparityCalculator = DP2DMultiBlocksDisparityCalculator(leftCens, rightCens);
    Mat_<int> disparity = disparityCalculator.computeDisparity(8);

//    calcHist()

    Mat_<float> ku(5, 5, 0.0), kv(5, 5, 0.0);
    ku(0, 0) = 2;
    ku(1, 0) = 2;
    ku(2, 0) = 4;
    ku(3, 0) = 2;
    ku(4, 0) = 2;
    ku(0, 1) = 1;
    ku(1, 1) = 1;
    ku(2, 1) = 2;
    ku(3, 1) = 1;
    ku(4, 1) = 1;
    ku(0, 4) = -2;
    ku(1, 4) = -2;
    ku(2, 4) = -4;
    ku(3, 4) = -2;
    ku(4, 4) = -2;
    ku(0, 3) = -1;
    ku(1, 3) = -1;
    ku(2, 3) = -2;
    ku(3, 3) = -1;
    ku(4, 3) = -1;

//    cout << ku << "\n";
//
    kv(0, 0) = 2;
    kv(0, 1) = 2;
    kv(0, 2) = 4;
    kv(0, 3) = 2;
    kv(0, 4) = 2;
    kv(1, 0) = 1;
    kv(1, 1) = 1;
    kv(1, 2) = 2;
    kv(1, 3) = 1;
    kv(1, 4) = 1;

    kv(3, 0) = -1;
    kv(3, 1) = -1;
    kv(3, 2) = -2;
    kv(3, 3) = -1;
    kv(3, 4) = -1;
    kv(4, 0) = -2;
    kv(4, 1) = -2;
    kv(4, 2) = -4;
    kv(4, 3) = -2;
    kv(4, 4) = -2;

//    cout << kv << "\n";
//
    auto conv = convolution(disparity, ku);

    auto conv_u = normalization(conv, ku);
    conv = convolution(disparity, kv);
    auto conv_v = normalization(conv, kv);

    cout << conv_u;

//    auto m = *min_element(conv_u.begin<float>(), conv_u.end<float>());
//    cout << m << "\n";
//    conv_u = conv_u - m;
//    m = *min_element(conv_v.begin<float>(), conv_v.end<float>());
//    cout << m << "\n";
//    conv_v = conv_v - m;
//
//    auto du = scaleImg(conv_u);
//    auto dv = scaleImg(conv_v);

    ofstream f1("hist.txt"), f2("histlog.txt");
    auto h = compute2dHist(conv_u, conv_v);
    auto hlog = convertToLg(h);
    f1 << h;
    f2 << hlog;

//
    namedWindow("h", WINDOW_NORMAL);
    namedWindow("left", WINDOW_NORMAL);
    namedWindow("dv", WINDOW_NORMAL);
    namedWindow("du", WINDOW_NORMAL);
    imshow("h", Mat_<uchar>(h));
    namedWindow("hlog", WINDOW_NORMAL);
    imshow("hlog", scaleImg(hlog));
    imshow("dv", conv_u);
    imshow("du", conv_v);
    imshow("left", left);

    Mat heatmap;
    applyColorMap(scaleImg(disparity), heatmap, COLORMAP_JET);
    namedWindow("disp", WINDOW_KEEPRATIO);
    namedWindow("dispGray", WINDOW_KEEPRATIO);
    imshow("disp", heatmap);
    imshow("dispGray", scaleImg(disparity));
    imwrite("disp.png", heatmap);
    imwrite("dispBlack.png", scaleImg(disparity));


    Mat a;
    Mat heatMapHSV;
    cvtColor(heatmap, heatMapHSV, COLOR_BGR2HSV);

    waitKey(0);
    return 0;
}
