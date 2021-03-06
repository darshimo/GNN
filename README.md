# GNN

## 概要

PFNサマーインターン2019のコーディング課題  
フルスクラッチでGraph Neural Networkを実装しました．

## 環境

g++ 8.3.0  
eigen

## ビルド方法

`make`コマンドを実行すると，task01 ~ task04 の実行ファイルがまとめて生成されます．  
`make task01`などとすれば，実行ファイルごとに生成されます．

## 実行手順

`./task01`で課題1の結果を標準出力に書き込みます．他の課題についても同様です．  
実行時に以下のオプションでパラメータを指定できます．指定しなかった場合，括弧内の値が設定されます．  
task03，task04はdatasetsがあるのと同じディレクトリに置いて実行してください．

### task01
* -v : グラフの頂点数(10)
* -d : 特徴ベクトルの次元(8)
* -t : 集約の回数(2)

ランダムに生成した隣接行列，これをもとに計算したh<sub>G</sub>を出力します．
確認用に，(集約-1)，(集約-2)，(READOUT)を愚直に計算して求めたh<sub>G</sub>も出力します．

### task02
* -v : グラフの頂点数(10)
* -d : 特徴ベクトルの次元(8)
* -t : 集約の回数(2)
* -l : 学習率(0.0001)

ランダムに生成した隣接行列とラベル，パラメータの更新による損失の変化を出力します．

### task03
* -a : 学習アルゴリズム(0)
* -d : 特徴ベクトルの次元(8)
* -t : 集約の回数(2)
* -l : 学習率(0.0001)
* -m : モーメント(0.9)
* -p : Adamのパラメータβ<sub>1</sub>(0.9)
* -q : Adamのパラメータβ<sub>2</sub>(0.999)
* -b : バッチサイズ(50)
* -e : エポック数(100)

オプション-aの引数が0ならSGD，1ならmomentumSGD，2ならAdamで学習します．

学習用データでの学習が1エポック分終わるごとに，学習用データ内での平均損失，平均精度を出力します．
学習がすべて終了した後，検定用データ内での平均損失，平均精度を出力します．

### task04
* -d : 特徴ベクトルの次元(8)
* -t : 集約の回数(2)
* -l : 学習率(0.001)
* -p : Adamのパラメータβ<sub>1</sub>(0.9)
* -q : Adamのパラメータβ<sub>2</sub>(0.999)
* -b : バッチサイズ(50)
* -e : エポック数(100)

task03と同様に，学習１エポックごとに学習用データ内での平均損失，平均精度を出力します．
その後，テスト用データに対して分類を行い，予測ラベルを`prediction.txt`に出力します．
