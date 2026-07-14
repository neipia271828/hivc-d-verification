# GPUサーバーでの新規実験実行手順

HIVC-D の Qwen 二者エピソードを GPU サーバーで実行し、ブラウザのライブビジュアライザーで確認するための手順です。

## 前提

- GPU サーバー: `student222@172.16.51.202`（SSH ポート `2222`）
- GPU 上の配置先: `~/projects/hivc-d-verification`
- ライブ可視化のポート: `8765`
- 実験設定: `configs/experiment.yaml`
- Python 環境: `~/projects/hivc-d-verification/.venv`

GPU 上の `~/projects/hivc-d-verification` は `.git` がない場合があります。その場合、GPU 上で `git pull` はできません。ローカルで可視化コードを変更したときは、Mac 側から `scripts/deploy_viz_to_gpu.sh` を使って配備してください。

実験用の依存パッケージは GPU 側の `.venv` に入っています。システムの `python3` には `numpy`、`torch`、`transformers` などが入っていないため、以後のコマンドでは必ず `.venv/bin/python` を使います。

`configs/experiment.yaml` の `model_path: ~/models/Qwen3-14B` は、実験ランナーがホームディレクトリを展開して使います。GPU側の実験スクリプトがこの対応より古い場合は、`--model-path /home/student222/models/Qwen3-14B` を明示してください。

## 1. 可視化サーバーを起動する

GPU サーバーで以下を実行します。

```bash
cd ~/projects/hivc-d-verification
.venv/bin/python scripts/live_server.py --port 8765 --ngrok
```

表示された `https://<ngrok-url>/visualize` をブラウザで開きます。

`OSError: [Errno 98] Address already in use` が出た場合は、すでに可視化サーバーが `8765` を使用しています。この起動で一時表示された ngrok URL は利用できません。既存サーバーを使うか、次のように再起動します。

```bash
lsof -nP -iTCP:8765 -sTCP:LISTEN
kill <表示されたPID>
.venv/bin/python scripts/live_server.py --port 8765 --ngrok
```

## 2. まず1ゲームだけ動作確認する

可視化サーバーとは別の GPU サーバーターミナルで、最小の確認実験を起動します。

```bash
cd ~/projects/hivc-d-verification

.venv/bin/python -u scripts/qwen_two_agent_experiment.py \
  --config configs/experiment.yaml \
  --conditions hivc_d \
  --games 1
```

実験開始時に `hivc_sim/results/turn_game/experiment/stream.jsonl` は空へリセットされます。モデル読み込み後、最初のターンが終了してから1行ずつ追記されます。ビジュアライザーを更新し、ライブモードで `stream.jsonl` を選択してください。

## 3. 全条件の本番実験をバックグラウンドで実行する

標準設定では `control`、`consulting`、`hivc_d` の各30ゲームを実行します。SSH 切断後にも処理を続けるため、`nohup` とログファイルを使います。

```bash
cd ~/projects/hivc-d-verification
mkdir -p logs

nohup .venv/bin/python -u scripts/qwen_two_agent_experiment.py \
  --config configs/experiment.yaml \
  > logs/experiment-$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 &
```

## 4. 進捗を確認する

```bash
# 実験プロセスの有無
ps -ef | grep '[q]wen_two_agent_experiment'

# 最新ログを追従
tail -f $(ls -t logs/experiment-*.log | head -1)

# ライブストリームの行数を確認
wc -l hivc_sim/results/turn_game/experiment/stream.jsonl
```

`stream.jsonl` が0バイトのままで、実験プロセスも存在しない場合は、最初のターンが記録される前に実験が停止しています。この場合は上記ログの末尾を確認してください。

## 5. 既存結果を上書きせずに新しい実験を残す

既定の出力先を使うと、条件別CSV、`all_games.csv`、`summary.csv`、ライブJSONLは上書きされます。実験ごとに分けて保存する場合は、タイムスタンプ付きの出力先を指定します。

```bash
cd ~/projects/hivc-d-verification
STAMP=$(date +%Y%m%d-%H%M%S)

.venv/bin/python -u scripts/qwen_two_agent_experiment.py \
  --config configs/experiment.yaml \
  --output-dir "hivc_sim/results/turn_game/experiment/$STAMP" \
  --live-jsonl "hivc_sim/results/turn_game/experiment/$STAMP/stream.jsonl"
```

ビジュアライザーの更新ボタンを押すと、新しい `stream.jsonl` がファイル一覧に現れます。新しいJSONLを選択してライブモードへ切り替えてください。
