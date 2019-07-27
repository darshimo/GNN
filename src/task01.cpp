#include <iostream>
#include <random>
#include <ctime>
#include <Eigen/Dense>
#include <cstdlib>
#include <unistd.h>
#include "net.hpp"

using namespace std;

int main(int argc, char *argv[]){
    srand((unsigned) time(NULL));


    int V = 10;//頂点の数
    int D = 8;//特徴ベクトルの次元
    int T = 2;//集約の回数


    int opt;
    opterr = 0;
    while((opt = getopt(argc, argv, "v:d:t:")) != -1){
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
            default:
                printf("Usage: %s [-vdt INT]\n", argv[0]);
                printf("\n");
                printf("  v INT : set the number of vertices\n");
                printf("  d INT : set the dimention of the feature vector\n");
                printf("  t INT : set the number of aggregation\n");
                exit(1);
        }
    }
    printf("V = %d\n",V);
    printf("D = %d\n",D);
    printf("T = %d\n",T);
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
    VectorXd h_G1 = ob.readout(G);
    cout << "h_G : " << endl;
    cout << h_G1 << "\n" <<  endl;


    //一致することを確認
    MatrixXd W = ob.get_W();
    VectorXd x[V], a[V];
    for(int v=0;v<V;v++){
        x[v] = VectorXd::Zero(D);
        x[v](0) = 1;
    }
    //T回ループ
    for(int t=0;t<T;t++){
        //集約1
        for(int v=0;v<V;v++){
            a[v] = VectorXd::Zero(D);
            for(int w=0;w<V;w++){
                if(N(v,w)==1){
                    a[v] += x[w];
                }
            }
        }
        //集約2
        for(int v=0;v<V;v++){
            x[v] = (W*a[v]).cwiseMax(0);
        }
    }
    //READOUT
    VectorXd h_G2 = VectorXd::Zero(D);
    for(int v=0;v<V;v++)h_G2 += x[v];

    cout << "h_G (test) : " << endl;
    cout << h_G2 << endl;

    return 0;
}
