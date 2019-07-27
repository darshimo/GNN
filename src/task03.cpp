#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <unistd.h>
#include <Eigen/Dense>
#include "net.hpp"
#include "sub.hpp"
#include "get_data.hpp"

using namespace std;

int main(int argc, char *argv[]){

    int algorithm = 0;//学習アルゴリズム
    int D = 8;//特徴ベクトルの次元
    int T = 2;//集約の回数
    double lr = 0.0001;//学習率
    double moment = 0.9;//モーメント
    double beta1 = 0.9;//Adamのパラメータ
    double beta2 = 0.999;//Adamのパラメータ
    int B = 50;//バッチサイズ
    int epoch = 100;//エポック数


    int opt;
    opterr = 0;
    while((opt = getopt(argc, argv, "a:d:t:l:m:p:q:b:e:")) != -1){
        switch(opt){
            case 'd':
                D = atoi(optarg);
                break;
            case 't':
                T = atoi(optarg);
                break;
            case 'l':
                lr = atof(optarg);
                break;
            case 'm':
                moment = atof(optarg);
                break;
            case 'p':
                beta1 = atof(optarg);
                break;
            case 'q':
                beta2 = atof(optarg);
                break;
            case 'b':
                B = atoi(optarg);
                break;
            case 'e':
                epoch = atoi(optarg);
                break;
            case 'a':
                algorithm = atoi(optarg);
                if(algorithm==0||algorithm==1||algorithm==2)break;
            default:
                printf("Usage: %s [-adtbe INT] [-lm DOUBLE]\n", argv[0]);
                printf("\n");
                printf("  a INT    : set the learning algorithm (0 : SGD, 1 : momentumSGD, 2 : Adam)\n");
                printf("  d INT    : set the dimention of the feature vector\n");
                printf("  t INT    : set the number of aggregation\n");
                printf("  b INT    : set the batch size\n");
                printf("  e INT    : set the number of epoch\n");
                printf("  l DOUBLE : set learning rate\n");
                printf("  m DOUBLE : set moment (parameter of momentumSGD)\n");
                printf("  p DOUBLE : set beta1 (parameter of Adam)\n");
                printf("  q DOUBLE : set beta2 (parameter of Adam)\n");
                exit(1);
        }
    }


    if(algorithm==0){
        printf("SGD\n");
    }else if(algorithm==1){
        printf("momentumSGD\n");
    }else{
        printf("Adam\n");
    }
    printf("D      : %d\n",D);
    printf("T      : %d\n",T);
    printf("lr     : %lf\n",lr);
    if(algorithm==1){
        printf("moment : %lf\n",moment);
    }else if(algorithm==2){
        printf("beta1  : %lf\n",beta1);
        printf("beta2  : %lf\n",beta2);
    }
    printf("B      : %d\n",B);
    printf("epoch  : %d\n",epoch);
    printf("\n");


    //データを取得
    int data_size = 2000;
    data *d_array = get_train_data_array(data_size);


    //学習用データと検定用データに分離
    int train_size, valid_size;
    data *train_data, *valid_data;
    train_size = split_data(d_array, data_size, &train_data, &valid_data, 0.7);
    valid_size = data_size - train_size;


    //学習
    net ob(D,T);
    if(algorithm==0){
        ob.SGD(train_data,train_size,lr,B,epoch);
    }else if(algorithm==1){
        ob.momentumSGD(train_data,train_size,lr,moment,B,epoch);
    }else{
        ob.Adam(train_data,train_size,lr,beta1,beta2,B,epoch);
    }
    //検定用データ内での平均損失と平均精度を計算
    double L = ob.loss(valid_data,valid_size);
    double prc = ob.precision(valid_data,valid_size);
    printf("\nvalidation\n");
    printf("loss : %2.6lf, precision : %2.6lf\n",L,prc);


    return 0;
}
