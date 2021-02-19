# GazePrediction
視線推定モデルの学習用コード

## 実験準備
### 実験環境
coiltraine-gazeの方に記載

### データセット
GazeDataを僕のHDDから用意

## プログラムの実行
### 学習コードの実行
aaa
#### オプション
- `--epoch`, `-e`:学習エポック数
- `--lr`, `-l`:学習率
- `--batch`, `-b`:バッチサイズ
- `--gpu-id`:使用するGPU番号
- `--train-dataset`, `-td`:学習データセットのパス
- `--valid-dataset`, `-vd`:評価データセットのパス
- `--exp-name`:実験名の指定．`experiments/`以下に実験名でディレクトリを作成，結果を保存
- `--model`, `-m`:学習する視線推定モデルの選択．選択肢は`co-conv`, `conv-cmd`, `conv-cmd-v2`
- `--dataset-type`, `-dt`:
- `--work-space`, `-ws`:

### 評価コードの実行
