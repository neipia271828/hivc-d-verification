# Devin API実験ガイド

## 位置づけ

この経路は Devin API v3 organization endpoints を使う新しい有料バックエンドである。GPUサーバー、Qwenモデル、`scripts/gpu_run.py`、Qwen実験ランナーには接続しない。Devin run はバックエンド条件が異なるため、過去のQwen/GPU runとの科学的な直接比較には使わない。

既定の smoke は `control / consulting / hivc_d` の3条件を各1ゲーム実行する。各 `(seed, condition)` に alpha と beta の独立セッションを1つずつ作るため、合計は **3条件 × 1ゲーム × 2エージェント = 6セッション**である。セッションは別ゲーム・別条件へ再利用しない。

## 認証と権限

認証は次の環境変数だけを使う。YAML、`.env`、コマンド引数、manifestには秘密情報を置かない。

- `DEVIN_API_KEY`: 必須。Devin API v3用credential。
- `DEVIN_ORG_ID`: 任意。未指定なら `GET /v3/self` の `org_id` を使い、取得できなければ明示的に失敗する。

実験には organization scope の `UseDevinSessions`、`ViewOrgSessions`、`ManageOrgSessions` 相当の権限が必要である。環境確認は安価な `GET /v3/self` だけを送る。

```bash
read -s "DEVIN_API_KEY?Devin API key: " && export DEVIN_API_KEY && echo
# 必要な場合だけ設定
export DEVIN_ORG_ID='org-...'

uv run devin-validate-env
```

出力されるのは principal種別、service user ID/name、organization ID、readinessだけである。API keyは表示、保存、ハッシュ化しない。

## 無料のdry-run

最初に必ずdry-runでセッション数、条件、seed、出力先を確認する。dry-runは環境変数を読まず、ネットワーク呼び出しも成果物作成も行わない。

```bash
uv run devin-experiment --dry-run
```

## 有料実験の実行

`configs/devin_experiment.yaml` は1セッション当たりとrun全体のACU上限、HTTP timeout/retry、poll間隔、応答timeout/retryを保守的に設定している。実行前にDevin側の料金とorganization残量も確認する。

```bash
uv run devin-experiment \
  --config configs/devin_experiment.yaml \
  --run-id devin-smoke-01
unset DEVIN_API_KEY
```

このコマンドは有料APIを呼ぶ。今回のバックエンド実装・検証では実行しない。

## セッション境界と応答契約

- 各ゲーム・条件で `alpha` と `beta` の2セッションを新規作成する。
- agentのprivate observation、Role、Persona、Valueを含むpromptは対応するセッションだけへ送る。
- 共通ゲームコードには話者付きprompt runnerを依存注入し、global monkeypatchは使わない。
- 各HTTP requestには `X-Request-ID`、各モデル要求には `request_correlation_id` を付ける。
- pollは保存済みcursorより後の新規messageだけを読み、`source=devin`、相関ID一致、厳密な単一JSON objectをすべて要求する。
- malformed/stale応答は設定上限内で同じセッションへ修復要求し、上限超過時は安全に失敗する。
- ゲーム終了時はセッションをterminateし、archiveを要求する。cleanup失敗は秘密を含まないHTTP provenanceに残る。

## 出力とローカルプレビュー

出力は `hivc_sim/results/turn_game/devin/runs/<run-id>/` に作る。

- `control_games.csv`, `consulting_games.csv`, `hivc_d_games.csv`
- `all_games.csv`, `summary.csv`
- `value_manifest.json`
- `manifest.json`: `backend=devin_api` とQwen/GPU非比較性を明記
- `devin_provenance.json`: session ID、status lifecycle、message event ID、latency、retry数、報告ACU、HTTP request IDを記録。private promptやcredentialは含めない

```bash
python3 scripts/local_preview.py \
  --downloads-dir hivc_sim/results/turn_game/devin/runs
# http://127.0.0.1:8765/
```

プレビューはローカルの `127.0.0.1` のみで行い、GPU側HTTP serverやngrokは使わない。
