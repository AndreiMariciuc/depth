//
// Created by andreimariciuc on 09.11.2022.
//

#ifndef DEPTHESTIMATION_UTILS_H
#define DEPTHESTIMATION_UTILS_H

//#define SHEAR_SCALE
//#define CREATE_VIDEO

typedef long long ll;
typedef long long img_t;

static double t;

#define COUNT_TIME(exec)       t = (double) getTickCount();     \
cout << "Starting process with :\n";                            \
exec;                                                           \
t = ((double) getTickCount() - t) / getTickFrequency();         \
printf("Time spent processing = %.3f [s]\n", t)                 \


inline bool isInside(int i, int j, int row, int cols) {
    return i >= 0 && i < row && j >= 0 && j < cols;
}

#endif //DEPTHESTIMATION_UTILS_H
