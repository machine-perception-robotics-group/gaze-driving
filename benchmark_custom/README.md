# Benchmark Custom
NoCrash Benchmarkにおいて，天候やエピソード，交通状況の条件を指定して実行する．
同時に走行時の中間特徴や推定視線マップを記録．

## カスタムベンチの実行
学習済みの実験が入ったディレクトリ（coiltraine-gaze）と実験を指定する．
<br>
以下サンプルコード

`python3 custom_drive.py --docker carlagear --save-features --gpus 0 -de NocrashCustomNewTown_Town02  -w 1 3 6 8 10 14 -ep 1 3 6 8 11 13 16 18 21 23 -tf all -cp 22000 -bf ../coiltraine --folder nocrash -e resnet34imnet10S1`

## カスタムベンチで保存した画像から可視化動画を作成
以下サンプルコード
