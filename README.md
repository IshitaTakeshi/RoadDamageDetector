## 実行環境


| Name         | Configuration                   |
|:-------------|:--------------------------------|
| OS           | Manjaro Linux                   |
| CPU          | Intel Core i7-6700              |
| RAM          | 64GB                            |
| GPU          | TITAN X (Pascal)  Memory 12GB   |
| GPU Driver   | nvidia-390                      |
| CUDA Version | 8.0                             |

## 環境設定

```
git clone https://github.com/PasonaTech-Inc/anomaly_detection.git
cd anomaly_detection
./setting-environment.sh
git checkout ssd300-vgg16-all-v0.1
```

## 学習

```
python3 train.py --gpu <gpu id>
```

## 実行

学習済みモデルを用いる場合はモデルファイル([link](https://drive.google.com/drive/folders/1yR-kJqier3tIOuawdS9b9OlJGEaIGF4y?usp=sharing))をダウンロードしておく．

```
python3 demo.py --gpu <gpu id> --pretrained_model models/all/ssd300-vgg16-v0.1/model.npz <path to image>
```

## 学習の設定

学習・実行用のコードはChainerCVの[SSD](https://github.com/chainer/chainercv/tree/master/examples/ssd)に基づいている．  
アーキテクチャはVGG16ベースのSSD300を用いている．

### データ
RoadDamageDatasetの全ての地区のデータをマージし，学習と評価に用いている．  
学習には各地区のtrainを統合したものを，評価には各地区のvalデータを統合したものを用いている．

### learning rate
初期値は5e-04としている．ExponentialShiftにより 80000 iteration 経過したときに5e-05，100000 iteration 経過したときに5e-06に変化する．

### 評価
学習時に 4000 iteration ごとに評価を行っている．評価基準はPascal VOCに準じており，各クラスにおけるAverage Precisionと全クラスに対する平均(mean Average Precision)を計測している．
