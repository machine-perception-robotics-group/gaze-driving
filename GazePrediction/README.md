# GazePrediction
視線推定モデルの学習用コード

## 実験準備
### 実験環境
coiltraine-gazeの方に記載

### データセット
GazeDataを僕のHDDから用意

## プログラムの実行
### 学習コードの実行
以下サンプルコード（あくまで自分の環境での）
`python3 gaze_model_train.py -b 16 --gpu-id 0 -td ../GazeData/Datasets/Gaze_average/GazeTrain/ -vd ../GazeData/Datasets/Gaze_average/GazeEval/ --exp-name co_vgg_dsv3_ave --model conditional_conv_model`

#### オプション
- `--epoch`, `-e`：学習エポック数
- `--lr`, `-l`：学習率
- `--batch`, `-b`：バッチサイズ
- `--gpu-id`：使用するGPU番号
- `--train-dataset`, `-td`：学習データセットのパス
- `--valid-dataset`, `-vd`：評価データセットのパス
- `--exp-name`：実験名の指定．`experiments/`以下に実験名でディレクトリを作成，結果を保存
- `--model`, `-m`：学習する視線推定モデルの選択．選択肢は`co-conv`, `conv-cmd`, `conv-cmd-v2`
- `--dataset-type`, `-dt`：使用するデータセットのタイプ．`v1`は単フレーム視線データ, `video`は動画視線データ
- `--work-space`, `-ws`：無視

### 評価コードの実行
以下サンプルコード（あくまで自分の環境での）
`python3 gaze_model_eval.py -m co-conv -b 1 -vd ../GazeData/Datasets/DriveGaze_Video/Val/ --test-epoch best_test --exp-name coconv_vid --gpu-id 0`
