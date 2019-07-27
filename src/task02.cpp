#include <iostream>
#include <cstdio>
#include <random>
#include <ctime>
#include <cstdlib>
#include <unistd.h>
#include <Eigen/Dense>
#include "net.hpp"
#include "param.hpp"

int main(int argc, char *argv[]){
    srand((unsigned) time(NULL));


    int V = 10;//頂点の数
    int D = 8;//特徴ベクトルの次元
    int T = 2;//集約の回数
    double lr = 0.0001;//学習率


    int opt;
    opterr = 0;
    while((opt = getopt(argc, argv, "v:d:t:l:")) != -1){
        switch(opt){
            case 'v':
                V = atoi(optarg);
                break;
            case 'd':
                D = atoi(optarg);
                break;
            case 't':
                T = atoi(optarg);
                break;
            case 'l':
                lr = atof(optarg);
                break;
            default:
                printf("Usage: %s [-vdt INT] [-l DOUBLE]\n", argv[0]);
                printf("\n");
                printf("  v INT    : set the number of vertices\n");
                printf("  d INT    : set the dimention of the feature vector\n");
                printf("  t INT    : set the number of aggregation\n");
                printf("  l DOUBLE : set learning rate\n");
                exit(1);
        }
    }
    printf("V  = %d\n",V);
    printf("D  = %d\n",D);
    printf("T  = %d\n",T);
    printf("lr = %lf\n",lr);
    printf("\n");


    //隣接行列を生成
    MatrixXd N(V,V);
    int tmp;
    for(int i=0;i<V;i++){
        for(int j=i;j<V;j++){
            tmp = rand() % 2;
            N(i,j) = tmp;
            N(j,i) = tmp;
        }
    }
    cout << "Adjacency matrix : " << endl;
    cout << N << "\n" << endl;


    //ネットワークにグラフを与えてh_Gを計算
    graph G;
    G.size = V;
    G.N = N;
    net ob(D,T);
    VectorXd h_G = ob.readout(G);

    int y_prd = ob.predict(G);

    data d;
    d.G = G;
    d.y = rand() % 2;
    cout << "label : " << d.y << "\n" << endl;

    //学習
    for(int i=0;i<100;i++){
        param grad = ob.gradient(&d,1);
        ob.update(grad*(-lr));
        double L = ob.loss(d);
        printf("%2d : %12.8lf\n",i,L);
    }


    return 0;
}
