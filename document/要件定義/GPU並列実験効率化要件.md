# GPU並列実験効率化要件

## 1. 目的

2枚のGPUをまたいで1モデルを実行する現行方式を見直し、各GPUに1モデル・1 workerを固定配置して、HIVC-Dの複数ゲーム実験を再現性と安全性を保ったまま並列化する。

効率化によって、`control`、`consulting`、`hivc_d` の条件定義、各条件に割り当てるseed、シナリオ、ペルソナ、生成パラメータ、評価指標を変更してはならない。

## 2. 背景と現状

2026-07-15に、RTX A5000 24GB×2のGPUサーバー上で、3条件×30ゲームの実験 `episode-20260715-210345` を計測した。

| 項目 | GPU0 | GPU1 |
|---|---:|---:|
| GPU | NVIDIA RTX A5000 | NVIDIA RTX A5000 |
| VRAM総容量 | 24,564 MiB | 24,564 MiB |
| 実験プロセスのVRAM使用量 | 約3,785 MiB | 約7,917 MiB |
| 12秒計測の主なGPU使用率 | 6〜10% | 33〜37% |
| 瞬間最大GPU使用率 | 76% | 100% |
| 温度 | 64〜66℃ | 81〜83℃ |
| 瞬間最大消費電力 | 約107W | 約197W |

現行の `load_model()` は `device_map="auto"` を用い、1つのQwen3-14B 4bitモデルを2GPUへ不均等に配置している。この構成では、次の問題がある。

- GPU0の平均利用率が低く、2枚分の計算能力を有効利用できていない。
- GPU1に計算・VRAM・温度が偏る。
- GPU間接続は `PHB` であり、モデル分割にPCIe経由のGPU間転送が伴う。
- 現行CLIは同時実験を一律で拒否するため、メモリに余裕があっても並列workerを起動できない。
- 計測時の処理速度は約7.2分/ゲームであり、90ゲームの単純推定は約11時間となる。ただし、この値は実行途中の少数ゲームによる暫定値とする。

## 3. スコープ

### 3.1 対象

- GPUごとの単一GPU worker起動
- 条件とseed範囲のshard分割
- 複数workerの実行・監視・失敗検出
- shard結果の整合性検査と結合
- GPU利用率、VRAM、温度、電力の記録
- 中断後の未完了shardのみの再実行
- GPU並列方式と現行方式のベンチマーク

### 3.2 非対象

- モデルの量子化方式や重みの変更
- ゲームルール、勝敗条件、評価関数の変更
- 1ゲーム内の複数エージェント生成をバッチ推論する最適化
- GPUサーバー上のHTTPサーバーや外部公開ビューアー
- 1GPU上で2 worker以上を定常運用する高密度実行

## 4. 設計原則

1. **1GPU・1worker**を標準構成とする。
2. workerは物理GPUを明示的に固定し、通常実行でモデルをGPU間分割しない。
3. 各条件に同じseed集合を与え、対応のある条件間比較を維持する。
4. GPU割当を条件と完全に固定しない。全条件のseedを両GPUへ同じ規則で割り当て、GPU差を条件効果と交絡させない。
5. shard単位で出力・終了コード・実行条件を独立保存する。
6. 集計値同士を平均せず、結合した生CSVから `summary.csv` を再計算する。
7. 並列度は速度ではなく、VRAM、温度、thermal throttle、エラー率を含む受入基準で決定する。
8. 進行中の正式runは、明示的な停止指示なしに並列化実装のベンチマーク対象にしない。

## 5. 実行アーキテクチャ

### 5.1 master runとshard

1回の実験全体を `master run`、各GPU workerが担当する連続seed範囲を `shard` とする。

```text
hivc_sim/results/turn_game/experiment/runs/<run-id>/
├── master_manifest.json
├── master.log
├── shards/
│   ├── <condition>-gpu0-seed42-56/
│   │   ├── shard_manifest.json
│   │   ├── run.log
│   │   ├── pid
│   │   ├── exit_code
│   │   └── <condition>_games.csv
│   └── <condition>-gpu1-seed57-71/
├── control_games.csv
├── consulting_games.csv
├── hivc_d_games.csv
├── all_games.csv
└── summary.csv
```

`master_manifest.json` は、master runの期待shard一覧と完了状態を保持する。`shard_manifest.json` は、個々のshardの実行条件と実行結果を保持する。

### 5.2 workerのGPU固定

- orchestratorはworkerごとに `CUDA_VISIBLE_DEVICES=<physical-gpu-id>` を設定する。
- workerから見えるGPUは1枚のみとし、モデルの全レイヤーをそのGPUへ配置する。
- 起動後に、モデルが複数の物理GPUへ配置されていないことを検査する。
- GPU ID、GPU UUID、GPU名、起動時VRAM使用量を `shard_manifest.json` へ保存する。
- 1GPU上へのモデル読込みがVRAM不足になる場合は自動的に量子化方式を変更せず、ベンチマーク失敗として明示する。

### 5.3 shard割当

`games=N`、基準seedを `S`、GPU数を `G`とする。

- 対象seed集合は `[S, S+N-1]` とする。
- seed集合をGPU数で可能な限り均等な連続範囲へ分割する。
- 割り切れない場合は、小さいGPU IDから1ゲームずつ多く割り当てる。
- 全条件で同じseed分割を使用する。
- `random_persona=true` の場合も、ペルソナ抽選は従来どおり各game seedから決定し、shard順序やGPU IDに依存させない。

30ゲーム、seed 42、2GPUの例は次のとおりとする。

| shard | GPU | games | seed |
|---|---:|---:|---|
| A | 0 | 15 | 42〜56 |
| B | 1 | 15 | 57〜71 |

この割当を `control`、`consulting`、`hivc_d` のすべてに適用する。「controlはGPU0、hivc_dはGPU1」のように条件でGPUを固定してはならない。

### 5.4 実行スケジュール

初期実装は、次のblocked scheduleを使用する。

1. `control` の2 shardをGPU0/GPU1で同時実行する。
2. 両shard完了後、`consulting` の2 shardを同時実行する。
3. 両shard完了後、`hivc_d` の2 shardを同時実行する。
4. 全6 shard完了後、整合性を検査して結果を結合する。

条件順序による時間帯の影響を評価する必要がある場合は、条件順序をrunごとに事前登録した順序へ入れ替える。同一run中で、空いたGPUへ別条件を場当たり的に割り当ててはならない。

## 6. CLI・設定要件

### 6.1 起動インターフェース

現行の `uv run experiment` とrun単位のログ保存を維持し、並列モードを明示的に指定できるようにする。

```bash
uv run experiment \
  --conditions control consulting hivc_d \
  --games 30 \
  --seed 42 \
  --gpus 0 1 \
  --parallel
```

最低限、次の引数を持つ。

| 引数 | 意味 | 既定値 |
|---|---|---|
| `--parallel` | shard並列モードを有効化 | false |
| `--gpus` | 使用を許可する物理GPU ID | 自動検出した利用可能GPU |
| `--workers-per-gpu` | GPUごとの最大worker数 | 1 |
| `--temperature-warning` | 警告温度 | 80℃ |
| `--temperature-stop-scheduling` | 新規shard起動を止める温度 | 83℃ |
| `--resume` | 成功済みshardを再利用して再開 | false |

`--workers-per-gpu 2` 以上は通常運用では禁止し、別途の高密度ベンチマークと明示的な安全基準の更新後にのみ許可する。

`--games N` は「1条件あたりの対応ありseed数」とする。例えば3条件で `--games 100`
なら、seedは100個であり、各seedを3条件で実行するため総ゲーム数は300となる。
manifestには `games_per_condition=100` と `total_condition_games=300` を別々記録し、seed数を
300に拡張してはならない。

### 6.2 事前検査

orchestratorは起動前に次を検査する。

- 指定GPU IDが存在し、compute modeで使用可能である。
- 他の未管理GPUプロセスと重複しない。
- 各GPUの空きVRAMが、事前ベンチマークで得たモデル読込みピークに安全係数を乗じた値以上である。初期安全係数は1.25とする。
- 温度が `temperature-stop-scheduling` 未満である。
- Git commit、実験config、フレームワーク、ペルソナファイルが確定している。
- 期待shard間で `(condition, seed)` が重複せず、各条件のseed集合が一致する。

事前検査に失敗した場合は、workerを1つも起動しない。

## 7. 温度・VRAM・負荷監視

### 7.1 監視項目

orchestratorは各GPUについて、少なくとも30秒ごとに次を記録する。

- timestamp
- GPU ID / UUID
- GPU utilization
- memory utilization
- VRAM used / total
- temperature
- power draw / power limit
- P-state
- thermal slowdown / hardware slowdownの有無
- worker PIDとshard ID

記録先は `gpu_metrics.csv` とし、master run直下に保存する。

### 7.2 安全動作

- 80℃以上で警告を記録する。
- 83℃以上では、実行中のゲームを即時強制終了せず、後続shardの新規起動を停止する。
- 83℃以上が60秒以上継続する、またはthermal slowdownが検出された場合、workerは現在のゲーム完了後に次のゲームへ進まず、shardを `paused_thermal` とする。
- CUDA out-of-memory、モデル読込失敗、GPUリセットは自動で並列度を上げる理由にせず、対象shardを失敗として保存する。
- モデルの自動再量子化、コンテキスト長の自動短縮、実験条件の自動変更は行わない。

## 8. 実験の公平性と再現性

### 8.1 固定する項目

同一master runの全shardで次を一致させる。

- Git commit SHA
- model pathとモデル識別子
- 量子化パラメータ
- 実験configの解決後本文とSHA-256
- フレームワーク本文、version、source、SHA-256
- `role.json`またはペルソナ入力のSHA-256
- max_new_tokens、thinking、議論・意思決定予算
- evaluator rolloutsと評価方策
- decision schedule seed
- 対象conditionとseed集合

### 8.2 GPU割当の交絡防止

- 各conditionにGPU0/GPU1の両方を割り当てる。
- 各GPUが担当するseed数は、condition間で同一とする。
- 主解析は対応するseedに基づき条件間を比較する。
- GPU IDとshard IDを全CSV行または結合可能なmanifestに記録し、必要に応じてGPU別の感度分析を行えるようにする。
- 単一GPU配置と現行の2GPU分割配置は数値計算経路が異なる可能性があるため、同一master run内で混在させない。

## 9. 結果結合

### 9.1 結合前検査

全shardが `completed` で、終了コードが0のときのみ正式結果を生成する。結合前に次を検査する。

- 各条件に `games` で指定した数の一意なseedがある。
- 各条件のseed集合が完全一致する。
- `(condition, seed, turn)` が重複しない。
- シナリオ、ペルソナ抽選、decision scheduleが同じseedで条件間一致する。
- 行スキーマと記録必須列が全shardで一致する。
- config、framework、persona、Git SHAのハッシュがmaster runの宣言値と一致する。

masterの `config_hash` はrun全体の不変設定を対象とし、orchestratorからworkerへ明示的に渡す。
workerが上書きするshard固有の `seed`、`games`、`output_dir` を含むhashは
`shard_config_hash` として別記録し、masterとの一致判定に使ってはならない。

一つでも失敗した場合は `summary.csv` を正式結果として生成せず、不足・重複・不一致の内容を `merge_report.json` へ記録する。
shardが `paused_thermal` または未完了の場合、完全性を前提とするseed集合・manifest結合検査は
`skipped: true` と記録し、空集合と全seedの差分を大量出力しない。部分CSVの行数は進捗情報として別記録する。

### 9.2 結合出力

- shardの行を `condition`、`seed`、`turn` の順に安定ソートする。
- 条件別CSVと `all_games.csv` を生成する。
- `summary.csv` は結合後の生行を `compute_summary_metrics()` に与えて再計算する。
- `merge_report.json` に入力shard、行数、seed数、検査結果、出力ファイルのSHA-256を保存する。
- 結合後もshard原本を削除・上書きしない。

## 10. 失敗・再開要件

- shardの状態は `pending`、`running`、`completed`、`failed`、`paused_thermal` のいずれかとする。
- workerが異常終了しても、他shardの原本と成功状態を変更しない。
- `--resume` は `completed` でハッシュが一致するshardを再実行せず、未完了または失敗shardのみを対象とする。
- `--resume` は同一run IDの既存 `master_manifest.json` を新manifestで上書きする前に読み、完了状態を引き継ぐ。
- run IDを省略した `--resume` は最後のrun IDを使い、新しいrunディレクトリを作成しない。
- 一部ゲームだけが出力されたshardは正常完了とみなさず、出力原本を保存したまま別のretry shardとして再実行する。
- retryで同じ `(condition, seed)` を二重採用しない。最終結合時に採用したshardをmanifestで明示する。
- 停止操作はmaster run全体とshard個別の両方をサポートする。

## 11. ベンチマーク計画

### 11.1 比較構成

正式導入前に、同じ実験configとseedを用いて次を比較する。

| 構成 | 内容 | 目的 |
|---|---|---|
| baseline | 現行の1 worker・2GPU `device_map=auto` | 現行速度と負荷の基準 |
| candidate | 2 worker・各1GPU固定 | 並列化後のスループットと安全性 |

- 各構成で少なくとも合4ゲームを実行する。
- モデル読込時間とゲーム実行時間を分離して計測する。
- 比較指標は games/hour、turns/hour、平均・p95ゲーム時間、GPU別利用率、ピークVRAM、最高温度、平均消費電力、失敗率とする。
- deterministic decodingであってもGPU配置の違いによる数値差の可能性があるため、モデル出力文の完全一致を受入条件にしない。
- JSON解析成功率、ゲーム完了率、行スキーマ、seed・シナリオ・ペルソナの割当一致は比較する。

### 11.2 採用基準

candidateは次のすべてを満たす場合に採用する。

1. baselineに対する games/hourが1.4倍以上である。
2. 各GPUのピークVRAM使用量が総VRAMの85%以下である。
3. CUDA out-of-memory、GPU reset、欠損ゲームが0件である。
4. thermal slowdownが発生しない。
5. 83℃以上が60秒を超えて継続しない。
6. 全ゲームの実験設定と割当がmanifestから復元できる。
7. 各条件のseed集合が同一で、欠損・重複がない。
8. 結合後の指標が生CSVから再計算される。

## 12. テスト要件

### 12.1 単体テスト

- NをGで割り切れる場合と割り切れない場合のshard分割
- 1ゲーム、2GPU指定のときに空shardを起動しないこと
- 条件ごとのseed集合同一性
- shard IDとパスのトラバーサル拒否
- GPU IDとworker PIDの対応
- 既存の実行中shardへの重複起動拒否
- 欠損、重複、ハッシュ不一致を含む結合失敗
- 結合後のsummary再計算
- `--resume` 時の成功済みshardスキップ
- 温度閾値と状態遷移

### 12.2 GPU統合テスト

1. GPU0だけで1ゲームを完了できる。
2. GPU1だけで1ゲームを完了できる。
3. GPU0/GPU1で同時に異なるseedを1ゲームずつ完了できる。
4. 同時実行中の `gpu_metrics.csv` が両GPUを記録する。
5. 両shard完了後、重複のない2ゲームとして結合できる。
6. 一方のshardを意図的に失敗させ、他方の原本を保ったまま正式集計を作らないことを確認する。

## 13. 完了条件

- 単一GPUへQwen3-14B 4bitを読み込み、1ゲームを完了できる。
- GPU0/GPU1の2 workerが異なるshardを同時実行できる。
- 条件ごとのseed集合とGPU割当が要件どおりである。
- 各workerのログ、終了コード、manifest、GPU指標が保存される。
- 失敗・温度上昇時に他shardの原本を損なわず、後続実行を停止または再開できる。
- 結合前検査を通過したときのみ、条件別CSV、`all_games.csv`、`summary.csv` を生成できる。
- baselineとcandidateのベンチマークレポートが残り、採用基準の合否を説明できる。
- 既存の単一run、GPUログ取得、ローカルプレビューの運用を破壊しない。

## 14. 導入順序

1. 稼働中の `episode-20260715-210345` を現行方式のbaselineとして完了させ、中断しない。
2. GPU指定、shard生成、manifest、結合・検査をローカルテスト可能な形で実装する。
3. GPU0、GPU1の単体スモークを順番に行う。
4. 2 worker同時の2ゲームスモークを行う。
5. baseline/candidateを各4ゲーム以上で計測する。
6. 採用基準を満たした場合のみ、正式実験の既定実行方式にする。
7. README、GPU実験手順、ローカルプレビューのrun認識をmaster run対応へ更新する。
