# 運転制御モデル学習＆評価用プログラム

### 導入方法
condaの仮想環境を利用．

基本的なはじめ方はオリジナルコードのGithubを参照 (https://github.com/felipecode/coiltraine

### データセットの用意
こちらから (https://github.com/felipecode/coiltraine/blob/master/docs/exploring_limitations.md

### シミュレータ環境
CARLA0.8.4をDocker上で使用．
導入方法はこちらの__Massive data collection__を参照 (https://github.com/felipecode/coiltraine/blob/master/docs/exploring_limitations.md

### 実行方法
#### 学習
```python3 coiltraine.py --single-process train --folder proposed_g1_vgg_norm_S1 -e myexpsS1 --gpus 0```

#### Validation
```python3 coiltraine.py --single-process validation --folder proposed_g1_vgg_norm_S1 -e myexpsS1 --gpus 0 -vd CoILVal```

#### ベンチマーク評価
```python3 coiltraine.py --gpus 0 --single-process drive -e proposed_g1_vgg_norm_S1 --folder myexpsS1 -de NocrashNewWeatherTown_Town02 --docker carlagear```

### 注意点
研究用にコードを書き換えたため，オリジナルのREADMEにあるような学習から評価をまとめて実行する方法は，おそらく動作しないから注意．
single-processを指定してバラバラに実行すること．
