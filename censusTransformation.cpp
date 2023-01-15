#include "censusTransformation.h"
#include "utils.h"
#include <thread>

const int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};
const int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};

const int S = 4;
const int T = 2;

void computeRows(const Mat_<uchar> &img, Mat_<unsigned short> &rimg, int ystart, int ystop) {
    ystop = min(ystop, img.rows);

    for (int y = ystart; y < ystop; y++) {
        for (int x = 0; x < img.cols; x++) {
            unsigned short r = 0;
            for (int k = 0; k < 8; k++) {
                int ii = y + S * dy[k];
                int jj = x + S * dx[k];
                if (isInside(ii, jj, img.rows, img.cols)) {
                    unsigned short xor1 = (img(y, x) - T) >= img(ii, jj) ? 0 : 1;
                    unsigned short xor2 = (img(y, x) + T) >= img(ii, jj) ? 2 : 3;
                    xor1 = xor1 ^ xor2;
                    r = r | (xor1 << (2 * k));
                }
            }
            rimg(y, x) = r;
//            std::cout << r << "\n";
        }
    }
}

template<typename T>
class CensusTransform {
public:
    std::vector<std::vector<T>> computeCensusTransform(const Mat_<uchar> &img) {
        std::vector<std::vector<T>> rez(img.rows, std::vector<ll>(img.cols, 0));

        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                rez[i][j] = computeDistance(i, j, img);
            }
        }

        return rez;
    }

private:
    T computeDistance(int i, int j, const Mat_<uchar> &img) {
        T r = 0;
        T one;
        int N = 8 * sizeof(T);

        if (N != 8) {
            for (int ii = 0; ii < 9; ii++) {
                for (int jj = 0; jj < 7; jj++) {
                    int ni = i + ii - 4;
                    int nj = j + jj - 3;
                    one = isInside(ni, nj, img.rows, img.cols) && img(i, j) < img(ni, nj) ? 1 : 0;
                    int idxShift = jj + ii * 9;
                    r |= one << idxShift;
                }
            }

            return r;
        }

        for (int k = 0; k < N; k++) {
            int ni = dy[k] + i;
            int nj = dx[k] + j;
            one = isInside(ni, nj, img.rows, img.cols) && img(i, j) < img(ni, nj) ? 1 : 0;
            r |= one << (N - 1 - k);
        }

        return r;
    }
};


std::vector<std::vector<ll>> censusTr(const Mat_<uchar> &img, int nbOfThreads) {
//    Mat_<unsigned short> rimg = Mat_<short>(img.rows, img.cols, (short) 0);
//
//    int dim = img.rows / nbOfThreads;
//
//    std::thread threads[nbOfThreads];
//
//    for (int i = 0; i < nbOfThreads; i++) {
//        threads[i] = std::thread(computeRows,
//                                 std::ref(img),
//                                 std::ref(rimg),
//                                 i * dim,
//                                 (i + 1) * dim);
//    }
//
//    for (auto &thread: threads) {
//        thread.join();
//    }
//
//    return rimg;

    CensusTransform<ll> censusTransform;

    return censusTransform.computeCensusTransform(img);
}
