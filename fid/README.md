# FID / KID 計算ツールの使い方

このディレクトリには FID（Fréchet Inception Distance）および KID（Kernel Inception Distance）を計算するためのユーティリティスクリプトが含まれます。主に次の 2 つのスクリプトを提供しています。

- `compute_cem_fid.py` : CEM 事前学習済み ResNet50（CEM500K / CEM1.5M）を特徴抽出に用いる EM 画像向けの FID/KID 計算スクリプト。
- `compute_normal_fid.py` : torchvision の ImageNet 学習済み Inception v3 を用いる、一般的な FID/KID 計算スクリプト。

以下に依存ライブラリ、各スクリプトの概要・主なオプション・使い方例、差分（前処理の違い）をまとめます。

## セットアップと実行フロー

### 1. 事前学習済み重み

- `fid/weights/` 配下に CEM500K / CEM1.5M のチェックポイントを配置してください。
- ダウンロードリンクやファイル名の詳細はリポジトリ直下の `./README.md` 「Pre-trained weights」節を参照してください。

### 2. 推奨実行形態

- **ローカルマシン**: Docker が利用できる環境では `run_fid_suite_docker.sh` の利用を推奨します。依存関係をビルド済みイメージに閉じ込められるため、最も再現性があります。
- **クラスタ / Docker 非対応環境**: `run_fid_suite_venv.sh` と Python venv を利用してください。GPU が使えるノードを選択すると計算が大幅に高速化します（CPU のみでも動作しますが、特徴抽出の所要時間が大きく増加します）。

### 3. 環境構築ヘルパースクリプト

`fid/setup_fid_env.sh` は Docker イメージのビルドと venv 構築を自動化します。

```bash
# Docker イメージのみビルド（ローカルデスクトップ向け）
./fid/setup_fid_env.sh --mode docker

# CUDA 対応 wheel を使って venv をセットアップ（クラスタ向け）
./fid/setup_fid_env.sh --mode venv \
  --venv-path /path/to/cem-fid-venv \
  --torch-index https://download.pytorch.org/whl/cu121
```

- 既定 `--mode auto` は Docker が見つかれば Docker イメージを、見つからなければ venv を構築します。`--mode both` で両方をまとめて準備できます。
- venv を構築した場合は、`run_fid_suite_venv.sh ... --venv /path/to/cem-fid-venv` のように同じパスを指定してください。
- Docker イメージは既定で `cem-fid` タグが付きます。別名にしたい場合は `--docker-tag` を指定してください。

### 4. 実行の流れ（まとめ）

1. 重みを `fid/weights/` に配置する。
2. `fid/setup_fid_env.sh` で Docker イメージまたは venv を準備する。
3. ローカルでは `run_fid_suite_docker.sh REAL_DIR GEN_DIR [OPTIONS] -- [EXTRA_ARGS]` を、クラスタでは `run_fid_suite_venv.sh REAL_DIR GEN_DIR --venv /path/to/venv ...` を実行する。
4. CEM-FID の結果 JSON は `fid/results/cem_fid/<backbone>/`（例: `fid/results/cem_fid/cem500k/`）に、通常 FID の結果は `fid/results/normal_fid/` に保存される。

> **GPU 利用推奨**: どちらのスクリプトも CPU のみで動作しますが、画像枚数が多い場合は GPU で実行した方が数十倍高速です。GPU が無い環境では `--batch-size` を小さくしてメモリ使用量を抑えてください。

## 共通の前提（依存ライブラリ）
両スクリプトで必要な Python パッケージ:

- `torch`
- `torchvision`
- `numpy`
- `scipy`
- `tqdm`

未インストールの場合は、仮想環境を有効にした上で次を実行してください。

```bash
pip install torch torchvision numpy scipy tqdm
```

## compute_cem_fid.py（CEM ResNet50 を使う）

概要
- `compute_cem_fid.py` は CEM500K (MoCoV2) または CEM1.5M (SwAV) の事前学習済み ResNet50 を特徴抽出器として用い、2 つの EM 画像フォルダ間の FID を算出します。オプションで KID も推定できます。グレースケールの EM 画像を自動で 3 チャンネルに変換し、CEM 事前学習時と同じ前処理（リサイズ・正規化）を適用した上で 2048 次元のグローバル平均プーリング特徴を抽出します。

基本的な使い方

```bash
python fid/compute_cem_fid.py REAL_DIR GEN_DIR [オプション]
```

- `REAL_DIR`: 実画像が入ったディレクトリ
- `GEN_DIR`: 生成画像が入ったディレクトリ

主なオプション（抜粋）

| オプション | 既定値 | 説明 |
|---|---:|---|
| `--backbone {cem500k, cem1.5m}` | `cem500k` | 使用する CEM 事前学習モデル |
| `--batch-size INT` | `32` | 特徴抽出時のバッチサイズ |
| `--num-workers INT` | `4` | DataLoader のワーカープロセス数 |
| `--device` | 自動 (GPU があれば `cuda`) | 推論デバイス |
| `--image-size INT` | `224` | 入力をリサイズするサイズ（CEM と同一） |
| `--weights-path PATH` | なし | 手動でダウンロードした重みを指定する場合に使用 |
| `--download-dir PATH` | なし | 重みのキャッシュ先を指定する場合に使用（`TORCH_HOME` を参照） |
| `--output-json PATH` | `cem_fid.json` | 結果保存先（実行時にタイムスタンプが付与される場合あり） |
| `--compute-kid` | 無効 | 指定すると KID も計算（特徴を保存して推定） |
| `--kid-subset-size INT` | `1000` | KID サブセットあたりのサンプル数 |
| `--kid-subset-count INT` | `100` | KID のサブセット試行回数 |
| `--seed INT` | `42` | KID 用乱数シード |

出力

- コンソールに FID（および KID の場合は平均と標準誤差）を表示します。
- 指定した `--output-json` に測定結果（FID/KID、バックボーン、画像数、正規化パラメータ、UTC タイムスタンプ、入力ディレクトリ等）を JSON として保存します。

備考

- 初回実行時に Zenodo から事前学習重みを自動ダウンロードする実装になっている場合があります（ネットワークが無い環境では手動ダウンロードして `--weights-path` を指定してください）。

## compute_normal_fid.py（ImageNet Inception v3 を使う）

概要
- `compute_normal_fid.py` は torchvision の ImageNet 学習済み Inception v3 を用いて、実画像群と生成画像群の FID（およびオプションで KID）を算出するユーティリティです。標準的な Inception ベースの FID 評価を行います。

基本的な使い方

```bash
python fid/compute_normal_fid.py REAL_DIR GEN_DIR [オプション]
```

主なオプション（抜粋）

| オプション | 既定値 | 説明 |
|---|---:|---|
| `--batch-size INT` | `32` | 特徴抽出時のバッチサイズ |
| `--num-workers INT` | `4` | DataLoader のワーカープロセス数 |
| `--device` | 自動 (GPU があれば `cuda`) | 推論デバイス |
| `--image-size INT` | `299` | Inception v3 が期待する入力解像度（既定 299） |
| `--output-json PATH` | `inception_fid.json` | 結果保存先（タイムスタンプ付きにする挙動あり） |
| `--data-volume STR` | なし | 実行環境メモ（ホスト:コンテナ マウント等）を記録する文字列 |
| `--compute-kid` | 無効 | 指定すると KID も計算（特徴を保存して推定） |
| `--kid-subset-size INT` | `1000` | KID サブセットあたりのサンプル数 |
| `--kid-subset-count INT` | `100` | KID のサブセット試行回数 |
| `--seed INT` | `42` | KID 用乱数シード |

出力

- コンソールに FID（および KID の場合は平均と標準誤差）を表示します。
- 指定した `--output-json` に測定結果（FID/KID、バックボーン名、重み情報、画像数、正規化値、UTC タイムスタンプ、入力ディレクトリ等）を JSON として保存します。

## 前処理の違い（両スクリプトの差分）

- `compute_cem_fid.py` は CEM 用の ResNet50 を使うため、グレースケール EM 画像を 3 チャンネル化し、CEM 事前学習時と同一の正規化（平均・標準偏差）および解像度（224）を用います。出力特徴は ResNet50 のグローバル平均プーリング（2048 次元）です。
- `compute_normal_fid.py` は ImageNet 学習済み Inception v3 を用いるため、RGB 入力（グレースケール→RGB に変換して使用可能）を 299×299 にリサイズし、ImageNet の正規化を適用します。Inception の出力を用いて FID/KID を計算します。

## ベストプラクティス

- 評価対象の画像のみを含むディレクトリを指定してください（スクリプトは再帰的に画像を探索します）。
- GPU 利用時は `--batch-size` を増やすと高速化できますが、メモリに注意してください。
- KID を有効にする場合、`--kid-subset-size` と `--kid-subset-count` の値により計算コストが増加します。適切な値に調整してください。

## Docker 利用例（CEM ResNet50 の例）

```bash
sudo docker run --rm \
  -v /path/to/real_and_fake:/data \
  -v /path/to/weights:/weights \
  -v /path/to/save/results:/results \
  cem-fid \
  /data/real /data/gen \
  --backbone cem500k \
  --weights-path /weights/cem500k_mocov2_resnet50_200ep.pth.tar \
  --output-json /results/cem_fid.json \
  --data-volume /path/to/real_and_fake:/data
```

## Docker ヘルパースクリプト（`run_fid_suite_docker.sh`）

リポジトリ同梱の `fid/run_fid_suite_docker.sh` を使うと、同じデータセットに対して CEM-FID と通常の Inception FID を連続で測定し、結果を `fid/results/` 配下へ保存できます。主な特徴:

- 事前に `fid/setup_fid_env.sh --mode docker` を実行して `cem-fid` イメージをビルドしてください。
- `REAL_DIR` と `GEN_DIR` をホスト側で指定すると、自動で `/data/real` / `/data/gen` にマウントして実行。
- `--cem-backbone {cem500k|cem1.5m}` を Script オプションとして指定するだけで、MoCoV2 (CEM500K) と SwAV (CEM1.5M) を切り替え可能（オプションを複数回指定すると、指定したすべてのバックボーンで順番に CEM-FID を計算します）。
- `fid/weights/` に配置したチェックポイントを自動検出（`--cem-weights` で明示的に指定することも可）。
- `--` 以降に書いた追加オプションは両方の Python スクリプトへ転送されます（例: `--batch-size 64`）。

SwAV 版で CEM-FID を計算したい場合の実行例:

```bash
./fid/run_fid_suite_docker.sh /path/to/real /path/to/gen \
  --cem-backbone cem1.5m \
  --cem-weights /home/tatsuki/デスクトップ/tatsuki_research/programs/cem-dataset/fid/weights/cem1.5m_swav_resnet50_200ep_balanced.pth.tar \
  -- --batch-size 64
```

上記では CEM-FID が SwAV バックボーンで計算され、続いて通常の Inception FID が同じデータで計測されます。`fid/results/cem_fid/cem1.5m/` と `fid/results/normal_fid/` にタイムスタンプ付き JSON が出力されるので、MoCoV2 との差分比較やログ管理が容易です。

MoCoV2 と SwAV の両方を一括で評価したい場合は、`--cem-backbone` を複数回指定してください。

```bash
./fid/run_fid_suite_docker.sh /path/to/real /path/to/gen \
  --cem-backbone cem500k \
  --cem-backbone cem1.5m \
  -- --batch-size 32
```

この例では CEM-FID が 2 回（cem500k → cem1.5m の順）実行され、それぞれ `fid/results/cem_fid/cem500k/` と `fid/results/cem_fid/cem1.5m/` に結果が保存されます。その後、通常の Inception FID が 1 回だけ実行され、`fid/results/normal_fid/` に保存されます。

### バッチ処理ヘルパー（`run_fid_suite_batch.py`）

同じ設定で複数のデータセットを一括評価したい場合は、JSON マニフェストを読み込む `fid/run_fid_suite_batch.py` が便利です。`jobs` 配列に複数のジョブを並べておくと、`run_fid_suite_docker.sh` が順番に実行されます。

```json
{
  "jobs": [
    {
      "name": "wannerfib-v2-v1",
      "real_dir": "/abs/path/to/WannerFIB/v2/real",
      "gen_dir": "/abs/path/to/WannerFIB/v1/gen",
      "cem_backbones": ["cem500k", "cem1.5m"],
      "extra_args": ["--batch-size", "32"]
    }
  ]
}
```

上記のようなファイルを `fid/batch_jobs.example.json` に用意しておき、次のように実行します。

```bash
./fid/run_fid_suite_batch.py fid/batch_jobs.example.json -- --batch-size 64
```

- `--script` で呼び出すスイートスクリプト（既定は `run_fid_suite_docker.sh`）を差し替え可能です。
- `jobs` エントリごとに `cem_backbones`, `script_args`（スイート側の追加 CLI オプション）, `extra_args`（`--` 以降に渡すオプション）を個別に設定できます。
- `--stop-on-error` や `--dry-run` などの制御オプションも用意しています。詳細は `fid/run_fid_suite_batch.py --help` を参照してください。

## venv ヘルパースクリプト（`run_fid_suite_venv.sh`）

Docker が利用できないクラスターでは、`fid/setup_fid_env.sh --mode venv --venv-path /path/to/venv` で依存ライブラリ入りの Python venv を用意し、以下のように実行します。

```bash
./fid/run_fid_suite_venv.sh REAL_DIR GEN_DIR \
  --venv /path/to/venv \
  --cem-backbone cem1.5m \
  -- --batch-size 64 --device cuda
```

- `--venv` を省略すると、`fid/../venv` または `fid/venv` を自動検出します。複数ユーザーで共有する場合はパスを明示してください。
- GPU ノードで実行する際は `--device cuda`（既定は自動判定）や `--batch-size` を環境に合わせて調整してください。CPU のみのノードでは実行できますが処理が非常に遅くなります。

## 追加の注意

- ネットワークが使えない環境では、事前学習重みを手動でダウンロードして `--weights-path` に渡してください。
- 各スクリプトは実行時のメタ情報（タイムスタンプ、入力ディレクトリ、使用オプションなど）を JSON に保存するため、実験ログとして利用できます。

---

スクリプト本体:

- `fid/compute_cem_fid.py`
- `fid/compute_normal_fid.py`

詳細は各ソースコード内のコマンドライン引数説明を参照してください。
