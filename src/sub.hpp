#ifndef SUB_H
#define SUB_H

#include <Eigen/Dense>

using namespace Eigen;
using namespace std;


struct graph{//グラフデータを保持
    int size;
    MatrixXd N;//隣接行列
};


struct data{//グラフデータとラベルの組を保持
    graph G;
    int y;
};


void shuffle(data *, int);
int split_data(data *, int, data **, data **, double ratio=0.7);

#endif
