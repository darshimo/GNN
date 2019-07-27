#ifndef NET_H
#define NET_H

#include <cstdio>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include "sub.hpp"
#include "param.hpp"

using namespace std;
using namespace Eigen;


class net{
    param theta;
    int D, T;
public:
    net(int,int);
    MatrixXd get_W();
    VectorXd readout(graph);
    int predict(graph);
    double loss(data);
    param gradient(data *, int);
    void update(param);
    double precision(data *, int);
    double loss(data *, int);
    void SGD(data *, int, double, int, int);
    void momentumSGD(data *, int, double, double, int, int);
    void Adam(data *, int, double, double, double, int, int);
};


net::net(int D, int T = 2){
    this->D = D;
    this->T = T;

    random_device seed_gen;
    default_random_engine engine(seed_gen());
    normal_distribution<> dist(0.0, 0.4);

    //パラメータの初期化
    theta.W = MatrixXd(D,D);
    theta.A = VectorXd(D);
    for(int i=0;i<D;i++){
        for(int j=0;j<D;j++){
            theta.W(i,j) = dist(engine);
        }
        theta.A(i) = dist(engine);
    }
    theta.b = 0;
}


MatrixXd net::get_W(){
    return theta.W;
}


VectorXd net::readout(graph G){
    int V = G.size;
    MatrixXd N = G.N;

    //特徴ベクトルの初期化
    MatrixXd X = MatrixXd::Zero(V,D);
    X.col(0) = VectorXd::Ones(V);

    //特徴ベクトルの更新をT回行う
    for(int t=0;t<T;t++){
        X = (N*X*theta.W.transpose()).cwiseMax(0);
    }

    //特徴ベクトルの合計
    VectorXd h_G = X.colwise().sum();

    return h_G;
}


int net::predict(graph G){//グラフデータからラベルを予測
    VectorXd h_G = readout(G);
    double s = theta.A.dot(h_G) + theta.b;
    int y_prd = s>0 ? 1 : 0;
    return y_prd;
}


double net::loss(data d){//グラフデータとラベルから損失を計算
    graph G = d.G;
    int y = d.y;
    VectorXd h_G = readout(G);
    double s = theta.A.dot(h_G) + theta.b;

    double L;
    if(s<-50){
        L = -y * s;
    }else if(s>50){
        L = (1.0 - y) * s;
    }else{
        L = y * log(1.0 + exp(-s)) + (1.0 - y) * log(1.0 + exp(s));
    }
    return L;
}


double net::loss(data *d_array, int d_size){//複数のグラフデータとラベルから平均損失を計算
    double sum = 0;
    for(int i=0;i<d_size;i++){
        data d = d_array[i];
        sum += loss(d);
    }
    double L = sum / d_size;
    return L;
}


param net::gradient(data *d_array, int d_size){
    double eps = 0.001;
    param grad(D);

    double L1 = loss(d_array, d_size);
    double L2;
    double tmp;

    //Wについて
    for(int i=0;i<D;i++){
        for(int j=0;j<D;j++){
            tmp = theta.W(i,j);
            theta.W(i,j) = tmp + eps;
            L2 = loss(d_array,d_size);
            grad.W(i,j) = (L2 - L1) / eps;
            theta.W(i,j) = tmp;
        }
    }
    //Aについて
    for(int i=0;i<D;i++){
        tmp = theta.A(i);
        theta.A(i) = tmp + eps;
        L2 = loss(d_array,d_size);
        grad.A(i) = (L2 - L1) / eps;
        theta.A(i) = tmp;
    }
    //bについて
    tmp = theta.b;
    theta.b = tmp + eps;
    L2 = loss(d_array,d_size);
    grad.b = (L2 - L1) / eps;
    theta.b = tmp;

    return grad;
}


void net::update(param update_amount){
    theta += update_amount;
    return;
}


double net::precision(data *d_array, int d_size){//複数のグラフデータとラベルから平均精度を計算
    double num_prd_0 = 0;//0と予測されたものの数
    double num_prd_1 = 0;//1と予測されたものの数
    double num_meet_0 = 0;//0と予測して正解した数
    double num_meet_1 = 0;//1と予測して正解した数
    for(int i=0;i<d_size;i++){
        data d = d_array[i];
        int y_prd = predict(d.G);
        if(y_prd==0){
            num_prd_0++;
            if(d.y==0)num_meet_0++;
        }else{
            num_prd_1++;
            if(d.y==1)num_meet_1++;
        }
    }
    double prc_0 = num_meet_0 / num_prd_0;
    double prc_1 = num_meet_1 / num_prd_1;
    double prc = (prc_0 + prc_1) / 2;
    return prc;
}


void net::SGD(data *d_array, int d_size,  double lr=0.0001, int B=50, int epoch=100){

    for(int e=0;e<epoch;e++){

        shuffle(d_array,d_size);

        for(int p=0;p<d_size;p+=B){
            //勾配を求めてパラメータを更新
            param grad = gradient(d_array+p, min(B, d_size-p));
            param update_amount = grad*(-lr);
            update(update_amount);
        }

        double L = loss(d_array, d_size);
        double prc = precision(d_array, d_size);
        printf("epoch : %2d, loss : %2.6lf, precision : %2.6lf\n",e,L,prc);

    }

    return;
}


void net::momentumSGD(data *d_array, int d_size, double lr=0.0001, double moment=0.9, int B=50, int epoch=100){
    param w(D);

    for(int e=0;e<epoch;e++){

        shuffle(d_array,d_size);

        for(int p=0;p<d_size;p+=B){
            //勾配を求めてパラメータを更新
            param grad = gradient(d_array+p, min(B, d_size-p));
            param update_amount = grad*(-lr) + w*moment;
            update(update_amount);
            w = update_amount;
        }

        double L = loss(d_array, d_size);
        double prc = precision(d_array, d_size);
        printf("epoch : %2d, loss : %2.6lf, precision : %2.6lf\n",e,L,prc);

    }

    return;
}


void net::Adam(data *d_array, int d_size, double lr=0.001, double beta1=0.9, double beta2=0.999 ,int B=50, int epoch=100){
    param m(D), v(D);
    double eps = 1e-8;
    int itr = 0;

    for(int e=0;e<epoch;e++){

        shuffle(d_array,d_size);

        for(int p=0;p<d_size;p+=B){
            //勾配を求めてパラメータを更新
            itr++;
            param grad = gradient(d_array+p, min(B, d_size-p));
            m += (grad - m) * (1.0 - beta1);
            v += (grad.cwiseAbs2() - v) * (1.0 - beta2);
            param update_amount = (m / (v.cwiseSqrt() + eps)) * (-lr * sqrt(1.0 - pow(beta2,itr)) / (1.0 - pow(beta1,itr)));
            update(update_amount);
        }

        double L = loss(d_array, d_size);
        double prc = precision(d_array, d_size);
        printf("epoch : %2d, loss : %2.6lf, precision : %2.6lf\n",e,L,prc);

    }

    return;
}


#endif
