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
# configファイルで実行（推奨・最も簡潔）
python3 scripts/qwen_two_agent_experiment.py --config configs/experiment.yaml

# CLI引数で個別項目を上書き（configより優先）
python3 scripts/qwen_two_agent_experiment.py --config configs/experiment.yaml --games 50
python3 scripts/qwen_two_agent_experiment.py --config configs/experiment.yaml --conditions control hivc_d

# スモークテストも同様
python3 scripts/qwen_turn_game_agent_smoke.py --config configs/agent_smoke.yaml
python3 scripts/qwen_two_agent_turn_game_smoke.py --config configs/smoke.yaml

# configなし（従来通り全引数をCLIで指定も可能）
python3 scripts/qwen_two_agent_experiment.py --model-path ~/models/Qwen3-14B --games 30
```

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
live_jsonl: null              # ライブ可視化用JSONLパス
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

### エージェント対話のグラフィカル可視化

`scripts/visualize_game.html` はブラウザでエージェント間の議論・投票・合意・状態遷移を可視化する。2つのモードがある。

**再生モード（CSVログから）**: ブラウザで `visualize_game.html` を開き、「CSVを読み込む」またはドラッグ＆ドロップで `all_games.csv` / `{condition}_games.csv` を選択する。条件タブ・ゲーム選択・タイムラインで任意のターンにジャンプでき、自動再生も可能。

**ライブモード（実験中リアルタイム）**: 実験スクリプトを `--live-jsonl` 付きで起動し、別途 `live_server.py` を立ててブラウザからポーリングする。

### リモートGPUサーバーから1コマンドで実験+可視化（推奨）

`scripts/gpu_run.py` は、このMacから1コマンドでGPUサーバー上の実験を起動し、ngrok経由でライブ可視化を自動開始する。

```bash
# フル自動: 実験起動 + サーバー起動 + ブラウザ自動オープン
python3 scripts/gpu_run.py

# 実験configを指定
python3 scripts/gpu_run.py --experiment-config configs/experiment.yaml

# 実験のみ（可視化なし）
python3 scripts/gpu_run.py --no-visualize

# サーバーのみ（実験は別途手動起動済みを想定）
python3 scripts/gpu_run.py --server-only

# 実験ログを tail -f で表示
python3 scripts/gpu_run.py --logs

# GPUサーバーの状態確認（プロセス・GPU使用率・ngrok）
python3 scripts/gpu_run.py --status

# 実験とサーバーを停止
python3 scripts/gpu_run.py --stop
```

事前準備:
1. `configs/gpu_server.yaml` にSSH接続情報を設定（`.gitignore`で除外済み）
2. GPUサーバーにngrokをインストール + `ngrok config add-authtoken`
3. GPUサーバーにコードを配置（`rsync` または `git push/pull`）

実行するとngrokの公開URLが表示され、ブラウザが自動で開く:

```
============================================================
  Live visualizer ready!
  Visualizer:  https://xxxx-xx-xx.ngrok-free.app/visualize
  JSONL stream: https://xxxx-xx-xx.ngrok-free.app/stream.jsonl
============================================================
```

### 手動でライブ可視化を起動する場合

SSH接続してGPUサーバー上で直接起動する場合:

```bash
# ターミナル1: 実験を実行（JSONLストリームを追記）
python3 scripts/qwen_two_agent_experiment.py \
  --config configs/experiment.yaml \
  --live-jsonl hivc_sim/results/turn_game/experiment/stream.jsonl

# ターミナル2: ライブ配信サーバー（ローカルアクセス）
python3 scripts/live_server.py \
  --file hivc_sim/results/turn_game/experiment/stream.jsonl \
  --port 8765

# ブラウザで http://localhost:8765/visualize を開き「ライブモード」ボタン →
# http://localhost:8765/stream.jsonl を指定
```

リモートGPUサーバーの場合は `--ngrok` オプションを追加:

```bash
python3 scripts/live_server.py --port 8765 --ngrok
```

または SSHポートフォワーディング（より安全）:

```bash
ssh -L 8765:localhost:8765 user@gpu-server
# ローカルPCのブラウザで http://localhost:8765/visualize にアクセス
```

可視化内容:
- 施設状態パネル（酸素・電力・船体損傷・浸水・通信・士気のバー、危険度色分け）
- イベントバッジ・勝敗結果
- エージェント議論のチャットバブル（α左・β右、行動タグ・理由・ready状態）
- 投票パネル（α票・グループ行動・β票・判定ルール・グループ理由）
- 評価パネル（最適行動・許容行動・regretゲージ・Q値バー）
- ペルソナ情報
- タイムライン（ターンごとのドット、勝敗色分け、クリックでジャンプ）

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
