#include <cstdio>
#include <cstdlib>
#include "sub.hpp"

using namespace std;


graph read_train_graph(int n){//datasets/train以下のn番目のグラフデータを取得
    graph G;
    char filename[50];
    sprintf(filename, "datasets/train/%d_graph.txt", n);
    FILE *fp;
    if((fp=fopen(filename,"r"))==NULL)exit(1);
    fscanf(fp, "%d",&G.size);
    G.N = MatrixXd(G.size,G.size);
    int tmp;
    for(int i=0;i<G.size;i++){
        for(int j=0;j<G.size;j++){
            fscanf(fp, "%d", &tmp);
            G.N(i,j) = tmp;
        }
    }
    fclose(fp);
    return G;
}


int read_train_label(int n){//datasets/train以下のn番目のラベルを取得
    int y;
    char filename[50];
    sprintf(filename, "datasets/train/%d_label.txt", n);
    FILE *fp;
    if((fp=fopen(filename,"r"))==NULL)exit(1);
    fscanf(fp, "%d",&y);
    fclose(fp);
    return y;
}


data read_train_data(int n){//datasets/train以下のn番目のグラフデータとラベルの組を取得
    data d;
    d.G = read_train_graph(n);
    d.y = read_train_label(n);
    return d;
}


graph read_test_graph(int n){//datasets/test以下のn番目のグラフデータとラベルの組を取得
    graph G;
    char filename[50];
    sprintf(filename, "datasets/test/%d_graph.txt", n);
    FILE *fp;
    if((fp=fopen(filename,"r"))==NULL)exit(1);
    fscanf(fp, "%d",&G.size);
    G.N = MatrixXd(G.size,G.size);
    int tmp;
    for(int i=0;i<G.size;i++){
        for(int j=0;j<G.size;j++){
            fscanf(fp, "%d", &tmp);
            G.N(i,j) = tmp;
        }
    }
    fclose(fp);
    return G;
}


data *get_train_data_array(int n){//datasets/train以下の0~n-1番目のデータの列を取得
    if(n>2000)exit(1);
    data *d_array = (data *)malloc(sizeof(data)*n);
    for(int i=0;i<n;i++){
        d_array[i] = read_train_data(i);
    }
    return d_array;
}


graph *get_test_graph_array(int n){//datasets/test以下の0~n-1番目のデータの列を取得
    if(n>500)exit(1);
    graph *g_array = (graph *)malloc(sizeof(graph)*n);
    for(int i=0;i<n;i++){
        g_array[i] = read_test_graph(i);
    }
    return g_array;
}
