#include <random>
#include <ctime>

#include "sub.hpp"


void shuffle(data *a, int size){//データの列をランダムに並べ替える
    srand((unsigned) time(NULL));
    for(int i=0; i<size; ++i){
        int idx = rand() % size;
        data t = a[i];
        a[i] = a[idx];
        a[idx] = t;
    }
    return;
}


int split_data(data *a, int size, data **train, data **valid, double ratio){//学習用データと検定用データを取得
    shuffle(a,size);
    int train_size = size * ratio;
    *train = a;
    *valid = a + train_size;
    return train_size;
}
