RoadDamageDetector in Chainer
=============================

Chainer版[RoadDamageDetector](https://github.com/sekilab/RoadDamageDetector)の実装

SSD(Single Shot Detector)を用いて道路損傷の検出を行うことができる．

詳細は以下を参照

* [Qiita](https://qiita.com/IshitaTakeshi/private/915de731d8081e711ae5)
* [GitHub wiki](github.com/IshitaTakeshi/anomaly_detection/wiki)

## 環境設定

依存パッケージのダウンロードやデータのダウンロード・展開は全て自動で行われる．

```
git clone https://github.com/PasonaTech-Inc/anomaly_detection.git
cd anomaly_detection
./setting-environment.sh
```

## 学習

```
python3 train.py --gpu <gpu id>
```

## 実行

学習済みモデルを用いる場合はモデルファイル([link](https://drive.google.com/drive/u/0/folders/1T_LwA8sjK_yoE7Z7Hv22Dz20G-GNxn1Z))をダウンロードしておく．

```
python3 demo.py --gpu <gpu id> --pretrained_model models/ssd300-vgg16-v0.1/model.npz <path to image>
```

### データ
データの詳細は[wiki](https://github.com/PasonaTech-Inc/anomaly_detection/wiki/Road-Damage-Dataset)に書かれている．  
RoadDamageDatasetの全ての地区のデータをマージし，学習と評価に用いている．  
学習には全地区のtrainをマージしたものを，評価には全地区のvalデータをマージしたものを用いている．
