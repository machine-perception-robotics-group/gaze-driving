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

`python3 custom_bench_video.py -e myexpsS1-2_proposed_gloss1_vgg_S1_22000_drive_control_output_NocrashCustomNewTown_Town02 -ep all -f feat0 feat1 feat2 feat3 gaze -mm -t`


## カスタムベンチにおける指定エピソードでの指定フレームを可視化
以下サンプルコード

`python3 custom_bench_vis.py -e nocrash_resnet34imnet10S1_22000_drive_control_output_NocrashCustom_Town01 -ep episode_3_2_134.49 -f feat0 feat1 feat2 feat3 -fn 453 -mm --no-blending`
