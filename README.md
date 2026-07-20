# HIVC-D 検証シミュレーション

対立解消フレームワーク **HIVC-D** の中核仮説——「共通目標の下でも判断基準 V の不一致が合意品質を低下させる」「HIVC-D 介入フローは単純交渉より優れる」——を、マルチエージェント・シミュレーションで定量検証するためのコードと結果一式。

研究の背景・設計・結果の詳細は **[PAPER.md](PAPER.md)**（日本語 / LaTeX版 [PAPER.tex](PAPER.tex)）を参照。
English version: **[PAPER_EN.md](PAPER_EN.md)** (LaTeX: [PAPER_EN.tex](PAPER_EN.tex)).

## 主要な知見（要約）

内的妥当性を確保した最終モデルにおいて、**V 不一致は合意ベクトルの C_t 乖離を引き起こさず（理論の第一リンクが不成立）、累積報酬にも影響しなかった**。これは「全選択肢が共通目標と整合し、合意が対称な平均であるレジームでは、V 不一致が平均操作によって自己相殺される」ことに起因する信頼できる帰無結果である。

## ディレクトリ構成

```
hivc_sim/
├── main.py          # エントリポイント・実験実行
├── config.py        # 全パラメータ定数
├── environment.py   # フィールド・餌・標識
├── agent.py         # 探索者モデル
├── agreement.py     # 合意メカニズム（条件A: baseline / 条件B: HIVC-D）
├── experiment.py    # 実験条件定義・試行管理
├── analysis.py      # 統計検定（H1〜H4）
├── visualize.py     # 軌跡・報酬曲線・ヒートマップ
├── grid_search.py   # ハイパーパラメータのグリッドサーチ
├── tests/           # pytest 単体テスト
└── results/         # 出力（figures / summary / raw / grid_search）
```

## セットアップ

Python 3.11+ を推奨。

```bash
pip install -r requirements.txt
```

## 実行方法

```bash
cd hivc_sim

# 単体テスト
python -m pytest tests/ -v

# 動作確認（10試行）
python main.py --trials 10 --seed 42

# 本実験（全600試行 = 3 V条件 × 2機構 × 100試行）
python main.py --trials 100

# ハイパーパラメータ・グリッドサーチ（36設定 × 30試行）
python grid_search.py --trials 30
```

### ターン制合意形成ゲーム

次フェーズ用の深海研究施設トラブルゲームは `turn_game.py` に実装している。LLM は行動選択だけを担当し、状態更新・イベント・勝敗判定・探索ベース評価関数はルールベースで処理する。

```bash
# 100ゲームをheuristic方策で自動プレイし、探索ベース評価ログをCSV出力
python3 turn_game_cli.py --games 100 --seed 42 --policy heuristic

# 比較用: random / heuristic / mcts
python3 turn_game_cli.py --games 100 --seed 42 --policy random
python3 turn_game_cli.py --games 100 --seed 42 --policy mcts
```

ターン制ゲームの出力先:
- `results/turn_game/*_games.csv` — ターン別ログ、状態、選択行動、Q値、regret
- `results/turn_game/*_summary.csv` — 勝率、生存率、平均スコア、平均regret、expert_match_rate、plan_revision_quality、minority_adoption_rate、conflict_resolution_quality（REQUIREMENTS §6）

### 事前ロールアウト検証と重み調整（REQUIREMENTS §5.2 / §8）

```bash
# 現行の終端スコア重みで §8 チェックレポートを出力
python3 hivc_sim/rollout_validation.py --games 100 --seed 42

# 重み調整も実行し、§8 制約を満たす win/loss 重みを探索
python3 hivc_sim/rollout_validation.py --games 100 --tune \
  --output hivc_sim/results/turn_game/validation.json
```

検証項目: 難易度が極端でない、heuristic > random、mcts >= heuristic、全行動が一定割合で最適、初期状態から勝利可能、イベントで最適行動が変化。

### LLM 実験ログと探索ベース評価ログの結合（REQUIREMENTS §10）

```bash
python3 hivc_sim/join_llm_eval.py \
  --llm-csv hivc_sim/results/turn_game/experiment/all_games.csv \
  --eval-csv hivc_sim/results/turn_game/heuristic_games.csv \
  --eval-label heuristic \
  --output hivc_sim/results/turn_game/experiment/joined_heuristic.csv \
  --summary hivc_sim/results/turn_game/experiment/joined_heuristic_summary.csv
```

`(seed, turn)` で LLM 判断と参照方策の行動・best_action・regret を横並びにし、結合後の §6 指標を summary に出力する。

### Qwen3-14B GPU環境

GPUサーバーでは、通常の依存に加えて `requirements-gpu.txt` を入れる。Qwen3-14B はBF16のままだと重いため、実験では4bit NF4量子化を標準にする。

```bash
pip install -r requirements-gpu.txt
python3 scripts/qwen3_14b_4bit_smoke.py --model-path ~/models/Qwen3-14B
```

### 設定ファイル（YAML）による実行

長いCLI引数をYAML設定ファイルに分離できる。`configs/` に既定値がある:

- `configs/experiment.yaml` — 3条件バッチ実験（§7）
- `configs/smoke.yaml` — 2エージェント対話スモーク（1ゲーム）
- `configs/agent_smoke.yaml` — 1エージェントスモーク（1ゲーム）

```bash
# configファイルで直接実行する場合は、runごとに新しいoutput-dirを指定する
python3 scripts/qwen_two_agent_experiment.py --config configs/experiment.yaml \
  --output-dir hivc_sim/results/turn_game/experiment/runs/episode-manual-01 \
  --run-id episode-manual-01

# CLI引数で個別項目を上書き（configより優先）
python3 scripts/qwen_two_agent_experiment.py --config configs/experiment.yaml --games 50 \
  --output-dir hivc_sim/results/turn_game/experiment/runs/episode-manual-02
python3 scripts/qwen_two_agent_experiment.py --config configs/experiment.yaml --conditions control hivc_d \
  --output-dir hivc_sim/results/turn_game/experiment/runs/episode-manual-03

# スモークテストも同様
python3 scripts/qwen_turn_game_agent_smoke.py --config configs/agent_smoke.yaml
python3 scripts/qwen_two_agent_turn_game_smoke.py --config configs/smoke.yaml

# configなし（従来通り全引数をCLIで指定も可能）
python3 scripts/qwen_two_agent_experiment.py --model-path ~/models/Qwen3-14B --games 30 \
  --output-dir hivc_sim/results/turn_game/experiment/runs/episode-manual-04
```

直列ランナーは既存run成果物や既存 `stream.jsonl` の再利用を拒否する。各runの `run_metadata.json` がCSV、stream、`value_manifest.json`、git commit、開始・完了状態を結び付ける。本実験前のsmoke科学的妥当性gateは [GPU実験実行手順](document/run-experiments/GPU実験実行手順.md) を参照。

configの主な項目（`configs/experiment.yaml`）:

```yaml
model_path: ~/models/Qwen3-14B
conditions: [control, consulting, hivc_d]
games: 30
seed: 42
max_new_tokens: 96
max_discussion_turns: 6
discussion_token_budget: 768
evaluator_rollouts: 24
output_dir: hivc_sim/results/turn_game/experiment
live_jsonl: null              # ライブ可視化は無効（ローカルCSVプレビューを使用）
role_file: role.json          # ペルソナ指定
alpha_role_key: null          # role.json内の任意キー
beta_role_key: null
random_persona: false         # true ならゲームごとにランダム選択
random_seed: null             # ランダムペルソナの抽選シード
```

### ランダムペルソナ指定

`random_persona: true` にすると、各ゲームごとに `role_file` から重複なしで2エージェントをランダム選択する。多様なペルソナ組み合わせでの実験に便利。

```yaml
# configs/experiment.yaml で
random_persona: true
random_seed: null             # null ならゲームごとの seed で異なる組み合わせ
```

```bash
# CLIで上書き
python3 scripts/qwen_two_agent_experiment.py --config configs/experiment.yaml --random-persona
```

- `random_seed: null`（既定）: ゲームごとの seed（`seed + game_index`）を使うため、ゲームごとに異なる組み合わせになる
- `random_seed: 42`（固定）: 全ゲームで同じペルソナ組み合わせになる（再現性用）
- alpha と beta は常に異なるエージェント（重複なし）

### 個別実行（CLI引数直接指定）

configを使わず全引数をCLIで指定することも可能:

```bash
# 1体エージェントでゲームを1回プレイ
python3 scripts/qwen_turn_game_agent_smoke.py --model-path ~/models/Qwen3-14B

# 2体エージェントが対話しながらゲームを1回プレイ
python3 scripts/qwen_two_agent_turn_game_smoke.py \
  --model-path ~/models/Qwen3-14B \
  --max-discussion-turns 6 \
  --discussion-token-budget 768

# ペルソナを直接指定
python3 scripts/qwen_two_agent_turn_game_smoke.py \
  --model-path ~/models/Qwen3-14B \
  --alpha-persona "安全管理担当。酸素、浸水、船体損傷を最優先する。" \
  --beta-persona "通信担当。通信復旧と電力維持を最優先する。"

# ペルソナをJSONから指定
python3 scripts/qwen_two_agent_turn_game_smoke.py \
  --model-path ~/models/Qwen3-14B \
  --personas-file personas/deepsea_default.json

# role.json の agent_01 / agent_02 を使う
python3 scripts/qwen_two_agent_turn_game_smoke.py \
  --model-path ~/models/Qwen3-14B \
  --role-file role.json

# role.json から任意の2エージェントを選ぶ
python3 scripts/qwen_two_agent_turn_game_smoke.py \
  --model-path ~/models/Qwen3-14B \
  --role-file role.json \
  --alpha-role-key agent_07 \
  --beta-role-key agent_24
```

### 3条件バッチ実験（REQUIREMENTS §7: control / consulting / hivc_d）

全条件に同一のゲームルール・初期状態・行動一覧を与え、条件間で差をつけるのは合意形成時に与える手順知識のみとする。モデルは1回だけロードし、条件 × N ゲームを反復する。

```bash
# configファイルで実行（推奨）
python3 scripts/qwen_two_agent_experiment.py --config configs/experiment.yaml

# CLIで上書き
python3 scripts/qwen_two_agent_experiment.py --config configs/experiment.yaml --games 50
```

出力先 (`output_dir`、既定 `hivc_sim/results/turn_game/experiment`):
- `{condition}_games.csv` — 条件別ターン別ログ（個人選択・個人理由・グループ理由・対立度込み、REQUIREMENTS §6）
- `all_games.csv` — 全条件結合
- `summary.csv` — 条件別 §6 主要評価指標

Q 値・best_action・acceptable_actions は議論中のエージェントには見せず、行動後の評価にのみ使う（REQUIREMENTS §7.1）。

### エージェント対話のグラフィカル可視化（ローカルオフライン）

実験結果は `scripts/download_gpu_logs.py` でローカルMacに取得し、`scripts/local_preview.py` でブラウザから可視化する。GPUサーバー上のngrokやライブサーバーは不要で、取得後はネットワークを切断しても閲覧できる。

#### uv統合CLI（推奨）

実験フローはリポジトリルートから次の4コマンドで実行できる。
詳しい操作方法、再実行、状態確認、停止、トラブル対応は **[GPU実験CLIの使い方](document/run-experiments/GPU実験実行手順.md)** を参照。

```bash
# 1. 変更をstage・commit・pushし、GPU側でgit pull
uv run sync

# 2. GPUで hivc_d 条件を1ゲーム実行（run ID・ログ・PIDを自動管理）
uv run experiment

# 3. 直前に開始した完了済みrunをMacへ取得
uv run download

# 4. 127.0.0.1:8765でGUIを起動し、ブラウザを開く
uv run visualize
```

`sync` は変更があれば `git add -A` とcommitを自動実行してからpushする。既定メッセージは `chore: sync experiment workflow` で、`--message "任意のメッセージ"` で変更できる。自動commitせず現在のHEADだけを同期する場合は `--allow-dirty` を指定する。GPU側に `.git` がない初回は、Macの `github-neipia` 鍵をSSH agent forwardingで一時利用してGit管理を復元する。秘密鍵はGPUへコピーせず、既存の `.venv` と実験結果も保持する。originが規定SSH URLと異なる、または同期後HEADが一致しない場合は安全のため停止する。

`experiment` は既定で `--conditions hivc_d --games 1` としてバックグラウンド起動する。runごとの出力はGPU側の `hivc_sim/results/turn_game/experiment/runs/<run-id>/` に保存され、`run.log`、`pid`、`exit_code`、CSV、`stream.jsonl` をまとめて管理する。`--parallel` 実行時は `shards/` 配下に `shard_manifest.json`、runルートに `master_manifest.json`、`gpu_metrics.csv`、`merge_report.json` が追加される。

```bash
# 任意条件・ゲーム数・seed
uv run experiment --conditions control consulting hivc_d --games 30 --seed 42

# GPU並列実行（1GPU・1worker、blocked schedule）
uv run experiment --parallel --gpus 0 1 --conditions control consulting hivc_d --games 30 --seed 42

# 状態・ログ・停止
uv run experiment --status
uv run experiment --logs
uv run experiment --stop

# run ID指定、既存ローカルrunの上書き
uv run download --run-id episode-20260714-120000 --overwrite

# ポート変更、ブラウザ自動起動なし
uv run visualize --port 8080 --no-open
```

```bash
# 1. GPUサーバーで実験を完了させる
python3 scripts/gpu_run.py

# 2. 完了したログをローカルに取得
python3 scripts/download_gpu_logs.py

# 3. ローカルプレビューサーバーを起動
python3 scripts/local_preview.py

# 4. ブラウザで http://127.0.0.1:8765/ を開く
```

`download_gpu_logs.py` は `configs/gpu_server.yaml` のSSH接続情報を使い、GPUサーバー上の `output_dir` を `hivc_sim/results/turn_game/downloads/<run-id>/` へ `rsync` する。取得時に `manifest.json` を生成し、どのGPU実験から来たかを追跡できる。

```bash
# 既定の実験出力を取得
python3 scripts/download_gpu_logs.py

# run名を指定
python3 scripts/download_gpu_logs.py --run-id 2026-07-13-run1

# リモート出力ディレクトリを変更
python3 scripts/download_gpu_logs.py --remote-output-dir hivc_sim/results/turn_game/experiment

# ドライラン
python3 scripts/download_gpu_logs.py --dry-run
```

`local_preview.py` は `127.0.0.1:8765` で待ち受け、以下を提供する:
- run選択
- 条件タブ（control / consulting / hivc_d）
- ゲーム選択（seed）
- ターン選択・自動再生
- 施設状態パネル（酸素・電力・船体損傷・浸水・通信・士気）
- イベントバッジ・勝敗結果
- エージェント議論のチャットバブル（α・β、行動タグ・理由・ready状態）
- 投票パネル（α票・グループ行動・β票・判定ルール・グループ理由）
- 評価パネル（最適行動・許容行動・regretゲージ・Q値バー）
- `summary.csv` の条件別集計

### リモートGPUサーバーから実験を起動

`scripts/gpu_run.py` は、このMacから1コマンドでGPUサーバー上の実験を起動する。

```bash
# デフォルトconfigで実験を起動
python3 scripts/gpu_run.py

# 実験configを指定
python3 scripts/gpu_run.py --experiment-config configs/experiment.yaml

# 実験ログを tail -f で表示
python3 scripts/gpu_run.py --logs

# GPUサーバーの状態確認（プロセス・GPU使用率）
python3 scripts/gpu_run.py --status

# 実験を停止
python3 scripts/gpu_run.py --stop
```

事前準備:
1. `configs/gpu_server.yaml` にSSH接続情報を設定（`.gitignore`で除外済み）
2. GPUサーバーにコードを配置（`git push/pull` または `rsync`）

実験が完了したら、上記の `download_gpu_logs.py` と `local_preview.py` でローカルに確認する。

ペルソナJSONは自然文だけでなく、以下のような構造化パラメータでも指定できる。

```json
{
  "alpha": {
    "role": "安全管理担当",
    "priority_weights": {
      "oxygen": 0.30,
      "power": 0.10,
      "hull_damage": 0.25,
      "flooding": 0.25,
      "communication": 0.10
    },
    "risk_tolerance": 0.20,
    "goal_focus": "survival_first",
    "communication_style": "cautious",
    "concession_tendency": 0.35,
    "evidence_demand": 0.70,
    "notes": "短期的な勝利よりも施設崩壊の回避を重視する。"
  }
}
```

`risk_tolerance`、`concession_tendency`、`evidence_demand` は 0.0-1.0 の値を使う。

既存シミュレーションの出力先:
- `results/raw/trials.csv` — 試行別生データ
- `results/summary/stats.csv` — 仮説検定の統計量
- `results/figures/` — 可視化（箱ひげ図・報酬曲線・軌跡・C_t 近似率・ρ 推移）
- `results/grid_search/` — グリッドサーチ結果

## 再現性

全乱数は `numpy.random.default_rng(seed)` で制御し、試行ごとに `seed = RANDOM_SEED_BASE + trial_id` を用いる。観測ノイズは探索者ごとに独立な RNG を割り当てる。

## ライセンス

[MIT License](LICENSE)
