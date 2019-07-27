#ifndef PARAM_H
#define PARAM_H

#include <Eigen/Dense>
#include <iostream>
#include <cmath>

using namespace Eigen;
using namespace std;


struct param{
    MatrixXd W;
    VectorXd A;
    double b;

    param(){};
    param(int);
    void print();
    param cwiseSqrt();
    param cwiseAbs2();
    param operator-();
    param operator+(const double);
    param operator*(const double);
    param operator/=(const double);
    param operator+(const param);
    param operator-(const param);
    param operator+=(const param);
    param operator/(const param);
};

param::param(int D){
    W = MatrixXd::Zero(D,D);
    A = VectorXd::Zero(D);
    b = 0;
}

void param::print(){//パラメータの出力
    cout << "W" << endl;
    cout << W << endl;
    cout << "A" << endl;
    cout << A << endl;
    cout << "b" << endl;
    cout << b << endl;
}

param param::cwiseSqrt(){//要素毎の平方根
    param tmp;
    tmp.W = W.cwiseSqrt();
    tmp.A = A.cwiseSqrt();
    tmp.b = sqrt(b);
    return tmp;
}
param param::cwiseAbs2(){//要素毎の平方
    param tmp;
    tmp.W = W.cwiseAbs2();
    tmp.A = A.cwiseAbs2();
    tmp.b = b*b;
    return tmp;
}

param param::operator-(){
    param tmp;
    tmp.W = -W;
    tmp.A = -A;
    tmp.b = -b;
    return tmp;
}

param param::operator+(const double x){
    param tmp;
    tmp.W = W.array() + x;
    tmp.A = A.array() + x;
    tmp.b = b + x;
    return tmp;
}
param param::operator*(const double x){
    param tmp;
    tmp.W = W * x;
    tmp.A = A * x;
    tmp.b = b * x;
    return tmp;
}
param param::operator/=(const double x){
    W /= x;
    A /= x;
    b /= x;
    return *this;
}

param param::operator+(const param right){
    param tmp;
    tmp.W = W + right.W;
    tmp.A = A + right.A;
    tmp.b = b + right.b;
    return tmp;
}
param param::operator-(const param right){
    param tmp;
    tmp.W = W - right.W;
    tmp.A = A - right.A;
    tmp.b = b - right.b;
    return tmp;
}
param param::operator+=(const param right){
    W += right.W;
    A += right.A;
    b += right.b;
    return *this;
}
param param::operator/(const param right){
    param tmp;
    tmp.W = (W.array() * right.W.cwiseInverse().array()).matrix();
    tmp.A = (A.array() * right.A.cwiseInverse().array()).matrix();
    tmp.b = b/right.b;
    return tmp;
}


#endif
