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

出力先:
- `results/raw/trials.csv` — 試行別生データ
- `results/summary/stats.csv` — 仮説検定の統計量
- `results/figures/` — 可視化（箱ひげ図・報酬曲線・軌跡・C_t 近似率・ρ 推移）
- `results/grid_search/` — グリッドサーチ結果

## 再現性

全乱数は `numpy.random.default_rng(seed)` で制御し、試行ごとに `seed = RANDOM_SEED_BASE + trial_id` を用いる。観測ノイズは探索者ごとに独立な RNG を割り当てる。

## ライセンス

[MIT License](LICENSE)
