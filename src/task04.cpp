#include <cstdio>
#include <cstdio>
#include <ctime>
#include <Eigen/Dense>
#include <unistd.h>
#include <cstdlib>
#include "net.hpp"
#include "sub.hpp"
#include "get_data.hpp"

using namespace std;

int main(int argc, char *argv[]){

    int D = 8;//特徴ベクトルの次元
    int T = 2;//集約の回数
    double lr = 0.001;//学習率
    double beta1 = 0.9;//Adamのパラメータ
    double beta2 = 0.999;//Adamのパラメータ
    int B = 50;//バッチサイズ
    int epoch = 100;//エポック数


    int opt;
    opterr = 0;
    while((opt = getopt(argc, argv, "d:t:l:p:q:b:e:")) != -1){
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
            default:
                printf("Usage: %s [-dtbe INT] [-lmv DOUBLE]\n", argv[0]);
                printf("\n");
                printf("  d INT    : set the dimention of the feature vector\n");
                printf("  t INT    : set the number of aggregation\n");
                printf("  b INT    : set the batch size\n");
                printf("  e INT    : set the number of epoch\n");
                printf("  l DOUBLE : set learning rate\n");
                printf("  p DOUBLE : set beta1 (parameter of Adam)\n");
                printf("  q DOUBLE : set beta2 (parameter of Adam)\n");
                exit(1);
        }
    }
    printf("Adam\n");
    printf("D     : %d\n",D);
    printf("T     : %d\n",T);
    printf("lr    : %lf\n",lr);
    printf("beta1 : %lf\n",beta1);
    printf("beta2 : %lf\n",beta2);
    printf("B     : %d\n",B);
    printf("epoch : %d\n",epoch);
    printf("\n");


    //学習用データとテスト用データを取得
    int d_size = 2000;
    int g_size = 500;
    data *d_array = get_train_data_array(d_size);
    graph *g_array = get_test_graph_array(g_size);


    //学習
    net ob(D,T);
    ob.Adam(d_array,d_size,lr,beta1,beta2,B,epoch);


    //テストデータに対する予測ラベルをファイルに出力
    FILE *fp;
    graph G;
    int y_prd;
    char filename[20] = "prediction.txt";
    if((fp=fopen(filename,"w"))==NULL)exit(1);
    for(int i=0;i<g_size;i++){
        G = g_array[i];
        y_prd = ob.predict(G);
        fprintf(fp,"%d\n",y_prd);
    }
    fclose(fp);


    return 0;
}
