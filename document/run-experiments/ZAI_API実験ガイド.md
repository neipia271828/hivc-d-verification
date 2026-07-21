# Z.ai GLM-4.7-Flash API実験ガイド

## 目的

GPUサーバーやDevin sessionを使わず、Z.ai一般推論APIの `glm-4.7-flash` で
`control / consulting / hivc_d` の二者実験を実行する。

この経路はモデルと推論サービスが異なるため、過去のQwen/GPU実験とは別実験として扱う。
条件間比較は同一run内の `glm-4.7-flash` 結果だけで行う。

## 認証

API keyは `ZAI_API_KEY`、`ZAI_API_KEY_2`、`ZAI_API_KEY_3` 環境変数から読む。
YAML、manifest、provenance、CSVには値を保存せず、環境変数名だけを記録する。

```bash
read -s "ZAI_API_KEY?Z.ai API key: " && export ZAI_API_KEY && echo
uv run zai-validate-env
```

`zai-validate-env` は一般推論endpointへ小さなJSON completionを1回送り、認証だけでなく
`glm-4.7-flash` とJSON modeが実際に利用可能か確認する。

使用するendpointは次の通りで、Coding Plan endpointではない。

```text
https://api.z.ai/api/paas/v4/chat/completions
```

## ネットワークなしの計画確認

```bash
uv run zai-experiment --dry-run
```

既定値は `configs/zai_experiment.yaml` の3条件×各1ゲームである。

## 3条件smoke

```bash
caffeinate -dimsu uv run zai-experiment \
  --config configs/zai_experiment.yaml \
  --parallel-conditions \
  --run-id zai-flash-smoke-01
```

smoke完了後、次を確認する。

- `manifest.json`: `status=completed`, `completed_game_conditions=3`
- `zai_provenance.json`: HTTP status、retry、条件・seed・agent別の実測token usage
- `summary.csv`: 3条件すべての集計
- 各条件CSV: JSON契約違反率、V測定、V*形成、最終投票

## 合計90ゲーム

smokeでJSON契約とrate limitに問題がないことを確認した後だけ実行する。

```bash
caffeinate -dimsu uv run zai-experiment \
  --config configs/zai_experiment_90games.yaml \
  --parallel-conditions \
  --run-id zai-flash-90games-01
```

設定は3条件×30 seed、合計90ゲームである。条件順は既存実装と同じくseedごとに決定論的に
ランダム化される。`--parallel-conditions` 指定時は同一seedの3条件を3workerで同時実行する。
3つの独立アカウントを使い、既定の `api_concurrency: 3 / api_concurrency_per_key: 1` では
各条件を別のkey slotで同時実行する。条件とkey slotの組合せはseedごとに循環し、30 seedでは
各条件が各アカウントを10回ずつ使う。要求間隔はkeyごとのadaptive limiterで制御し、通常5〜8秒、
`1302 / 1303 / 1305` 発生時は120秒cooldownし、最大30秒まで間隔を増やす。
`thinking_enabled: false` により、短いJSON契約へ不要なthinking tokenを使わない。

Macの蓋閉じ中はHTTP request、timeout、retry時刻が歪むため、実験中は蓋を開いたままにする。
コマンドは必ず `caffeinate -dimsu` 配下で実行する。clamshell sleepが混入したrunは速度・rate limit評価に使わない。

`compact_prompts: true` は、Role、観測状態、会話履歴、JSON契約を維持したまま、consulting/HIVC-Dの
固定手順と重複表現だけを短縮する。出力上限は `max_new_tokens: 256` とする。compact-v1は従来の
full prompt runとprompt条件が異なるため、別実験系列として扱う。
各 `(seed, condition)` の完了後にCSV、manifest、value manifest、provenanceを
チェックポイント保存するため、API制限で停止しても完了済み結果を失わない。

`max_total_tokens` はrun全体の安全上限である。APIが返した `usage.total_tokens` の累積が
上限を超えるとrunを失敗状態にして停止する。

## レート制限

HTTP 429、5xx、および一時的な高頻度・混雑系API codeは共有adaptive limiterで再試行する。
日次上限など再試行しても解消しない制限では、runを `failed` として終了し、最後の
チェックポイントを残す。

既定値は次のとおり。

```yaml
adaptive_rate_limit: true
adaptive_min_interval_seconds: 5.0
adaptive_initial_interval_seconds: 8.0
adaptive_max_interval_seconds: 30.0
adaptive_cooldown_seconds: 120
adaptive_max_retries: 20
```

## ローカルプレビュー

```bash
python3 scripts/local_preview.py \
  --downloads-dir hivc_sim/results/turn_game/zai/runs
```

ブラウザでは `http://127.0.0.1:8765/` を開く。外部公開は行わない。

## 終了時

```bash
unset ZAI_API_KEY
unset ZAI_API_KEY_2
unset ZAI_API_KEY_3
```
